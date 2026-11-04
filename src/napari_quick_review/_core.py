"""Parts of the reviewer that do not need a window: reading boxes, saving calls, writing training patches."""
import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd

# the three calls; the key is fixed so that exporting knows what each call means
CALLS = {"1": "correct", "2": "fix outline", "3": "false detection"}


def read_crop(data, center, half):
    """Cut a box around center (z, y, x) from a 3D array, clipped at the image edge.

    data can be a numpy, zarr or dask array; only the box is read from disk.
    Returns the crop and the position of its first voxel in the full image.
    """
    lo = [max(int(c) - h, 0) for c, h in zip(center, half)]
    hi = [min(int(c) + h + 1, n) for c, h, n in zip(center, half, data.shape)]
    crop = np.asarray(data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
    return crop, np.array(lo)


def object_centers(labels):
    """One row per object in a label image: its id and a voxel inside it (z, y, x).

    The voxel is the one closest to the object's center of mass, so the ring always sits on the
    object, also for curved objects whose center lies outside them. This reads the whole label
    image once.
    """
    from scipy import ndimage as ndi
    lab = np.asarray(labels)
    rows = []
    for i, box in enumerate(ndi.find_objects(lab), start=1):
        if box is None:
            continue
        pts = np.argwhere(lab[box] == i)
        p = pts[np.argmin(((pts - pts.mean(0)) ** 2).sum(1))]
        rows.append(dict(id=str(i), z=box[0].start + p[0], y=box[1].start + p[1], x=box[2].start + p[2]))
    return pd.DataFrame(rows, columns=["id", "z", "y", "x"])


def stratified_sample(positions, n, shape, bins=3, seed=0):
    """Choose n positions spread through the volume.

    The volume is cut into bins x bins x bins blocks (depth, height, width) and positions are taken
    from each block in turn, in random order within a block, so every part of the volume is reviewed.
    Blocks with few objects simply run out earlier. A fixed seed gives the same choice every time.
    """
    if n <= 0 or n >= len(positions):
        return positions.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    p = positions.reset_index(drop=True)
    block = [np.minimum((p[a].to_numpy() * bins // s).astype(int), bins - 1) for a, s in zip("zyx", shape)]
    key = block[0] * bins * bins + block[1] * bins + block[2]
    groups = [list(rng.permutation(np.flatnonzero(key == k))) for k in rng.permutation(np.unique(key))]
    order = []
    while len(order) < n:
        for g in groups:
            if g and len(order) < n:
                order.append(g.pop())
    return p.iloc[order].reset_index(drop=True)


class Calls:
    """The calls made so far, kept in a CSV that is rewritten after every key press.

    One row per object (id, call, key, time). Changing a call replaces the old row,
    so the file always holds the latest call for each object and a review can be resumed.
    """

    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            self.table = pd.read_csv(self.path, dtype={"id": str, "key": str})
        else:
            self.table = pd.DataFrame(columns=["id", "call", "key", "time"])

    def get(self, id_):
        row = self.table[self.table.id == str(id_)]
        return None if row.empty else row.call.iloc[-1]

    def done(self):
        return set(self.table.id)

    def save(self, id_, key):
        now = datetime.datetime.now().isoformat(timespec="seconds")
        new = pd.DataFrame([dict(id=str(id_), call=CALLS[key], key=key, time=now)])
        self.table = pd.concat([self.table[self.table.id != str(id_)], new], ignore_index=True)
        tmp = self.path.with_suffix(".tmp")
        self.table.to_csv(tmp, index=False)
        tmp.replace(self.path)          # write then rename, so a crash never leaves half a file


# ------------------------------------------------------------------------- training patches
def patch2d(data, z, y, x, size):
    """size x size patch of slice z centered on (y, x); zero-padded where it runs off the image."""
    out = np.zeros((size, size), dtype=data.dtype)
    y0, x0 = int(y) - size // 2, int(x) - size // 2
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + size, data.shape[1]), min(x0 + size, data.shape[2])
    out[ys - y0:ye - y0, xs - x0:xe - x0] = np.asarray(data[int(z), ys:ye, xs:xe])
    return out


def to_8bit(img, window):
    """empanada trains on 8-bit images, so other images are mapped to 0-255 with one fixed gray window
    (low, high) for the whole volume. The same mapping must be used for the images the model is run on."""
    if img.dtype == np.uint8:
        return img
    lo, hi = window
    return np.clip(np.round((img.astype(np.float32) - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)


def load_edits(edit_dir):
    """Outlines painted in the reviewer: one file per object, the edited label box and its corner."""
    edits = []
    for f in sorted(Path(edit_dir).glob("*.npz")) if Path(edit_dir).exists() else []:
        d = np.load(f)
        edits.append((str(f.stem), d["lab"], d["lo"]))
    return edits


def export_patches(image, labels, positions, calls, missed, out_dir, size=256, window=None, name="review",
                   edits=()):
    """Write the review as empanada training patches (2D, one slice through each object).

    Every mask is the model output in that patch, with outlines painted in the reviewer put in
    and objects called "false detection" removed.
      ready/          patches that can be used for training as they are
      needs_editing/  patches with an outline still to fix or a missed object to draw; correct the
                      mask in napari and save it, then they can be used too
    Returns the number of patches written to each folder.
    """
    out_dir = Path(out_dir)
    pos = positions.set_index("id")
    call = dict(zip(calls.id.astype(str), calls.call))

    def label_of(id_):
        r = pos.loc[id_]
        v = int(np.asarray(labels[int(r.z), int(r.y), int(r.x)]))
        return v if v else (int(id_) if str(id_).isdigit() else 0)

    false_ids = {label_of(i) for i, c in call.items() if c == "false detection"} - {0}
    edited = {e[0] for e in edits}
    fix_ids = {label_of(i) for i, c in call.items() if c == "fix outline" and i not in edited} - {0}
    if window is None and np.asarray(image[0:1, 0:1, 0:1]).dtype != np.uint8:
        raise ValueError("A gray window (low, high) is needed to turn a 16-bit image into 8 bit.")

    jobs = [(f"{name}_{i}", pos.loc[i], c) for i, c in call.items() if i in pos.index]
    jobs += [(f"{name}_missed{k + 1}", r, "missed") for k, r in enumerate(missed.itertuples())]
    counts = {"ready": 0, "needs_editing": 0}
    for fname, r, c in jobs:
        img = to_8bit(patch2d(image, r.z, r.y, r.x, size), window)
        msk = patch2d(labels, r.z, r.y, r.x, size).astype(np.int32)
        y0, x0 = int(r.y) - size // 2, int(r.x) - size // 2
        for _, lab, lo in edits:                       # paste painted outlines over the model output
            dz = int(r.z) - lo[0]
            if 0 <= dz < lab.shape[0]:
                ys, xs = max(lo[1], y0), max(lo[2], x0)
                ye, xe = min(lo[1] + lab.shape[1], y0 + size), min(lo[2] + lab.shape[2], x0 + size)
                if ye > ys and xe > xs:
                    msk[ys - y0:ye - y0, xs - x0:xe - x0] = lab[dz, ys - lo[1]:ye - lo[1], xs - lo[2]:xe - lo[2]]
        msk[np.isin(msk, list(false_ids))] = 0
        # a patch needs editing if it holds an outline still to fix or a missed object
        click_inside = ((missed.z == int(r.z)) & missed.y.between(y0, y0 + size - 1)
                        & missed.x.between(x0, x0 + size - 1)).any()
        unfixed = c == "missed" or (c == "fix outline" and fname[len(name) + 1:] not in edited)
        edit = unfixed or bool(np.isin(msk, list(fix_ids)).any()) or bool(click_inside)
        sub = out_dir / ("needs_editing" if edit else "ready") / name
        os.makedirs(sub / "images", exist_ok=True)
        os.makedirs(sub / "masks", exist_ok=True)
        import tifffile
        tifffile.imwrite(sub / "images" / f"{fname}.tif", img)
        tifffile.imwrite(sub / "masks" / f"{fname}.tif", msk)
        counts["needs_editing" if edit else "ready"] += 1
    if window is not None:
        (out_dir / "gray_window.txt").write_text(f"low {window[0]}\nhigh {window[1]}\n")
    return counts
