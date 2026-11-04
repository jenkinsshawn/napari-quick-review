import numpy as np
import pandas as pd
import tifffile

from napari_quick_review._core import (Calls, export_patches, load_edits, object_centers, read_crop,
                                       stratified_sample, to_8bit)


def test_read_crop_clips_at_edge():
    a = np.arange(10 * 20 * 30, dtype=np.uint16).reshape(10, 20, 30)
    crop, lo = read_crop(a, (1, 10, 28), (2, 3, 3))
    assert crop.shape == (4, 7, 5) and list(lo) == [0, 7, 25]
    assert crop[0, 0, 0] == a[0, 7, 25]
    assert crop.dtype == np.uint16             # 16-bit stays 16-bit


def test_calls_replace_and_resume(tmp_path):
    f = tmp_path / "calls.csv"
    c = Calls(f)
    c.save("a", "1")
    c.save("b", "2")
    c.save("a", "3")                           # changed call replaces the old row
    again = Calls(f)
    assert again.get("a") == "false detection" and again.done() == {"a", "b"}
    assert len(pd.read_csv(f)) == 2


def test_object_centers():
    lab = np.zeros((10, 10, 10), np.int32)
    lab[2:5, 2:5, 2:5] = 1
    lab[6:9, 0:2, 7:10] = 3                    # id 2 absent: skipped
    c = object_centers(lab)
    assert list(c.id) == ["1", "3"] and c.iloc[0][["z", "y", "x"]].tolist() == [3, 3, 3]


def test_object_centers_inside_curved_object():
    lab = np.zeros((1, 20, 20), np.int32)
    lab[0, 2, 2:18] = 1                        # an L shape: its box center (0, 9, 9) is outside it
    lab[0, 2:18, 2] = 1
    r = object_centers(lab).iloc[0]
    assert lab[r.z, r.y, r.x] == 1


def test_stratified_sample_covers_the_volume():
    rng = np.random.default_rng(1)
    # 900 objects crowded in one corner, 27 spread out: a plain random pick would mostly take the corner
    crowd = pd.DataFrame(dict(z=rng.integers(0, 30, 900), y=rng.integers(0, 30, 900), x=rng.integers(0, 30, 900)))
    g = np.arange(27)
    spread = pd.DataFrame(dict(z=(g // 9) * 33 + 16, y=((g // 3) % 3) * 33 + 16, x=(g % 3) * 33 + 16))
    pos = pd.concat([crowd, spread], ignore_index=True)
    pos["id"] = pos.index.astype(str)
    s = stratified_sample(pos, 27, (100, 100, 100))
    blocks = {tuple(v) for v in (s[["z", "y", "x"]].to_numpy() * 3 // 100)}
    assert len(s) == 27 and len(blocks) == 27   # one object from each of the 27 blocks
    assert stratified_sample(pos, 27, (100, 100, 100)).id.tolist() == s.id.tolist()   # same choice again


def test_to_8bit_window():
    a = np.array([[1000, 2000, 3000]], np.uint16)
    assert to_8bit(a, (1000, 3000)).tolist() == [[0, 128, 255]]


def test_export(tmp_path):
    img = np.full((5, 600, 600), 2000, np.uint16)
    lab = np.zeros(img.shape, np.int32)
    lab[2, 100:110, 100:110] = 1               # correct
    lab[2, 140:150, 140:150] = 2               # false detection, inside object 1's patch
    lab[2, 500:510, 500:510] = 3               # fix outline, painted below
    lab[2, 300:310, 450:460] = 4               # fix outline, not painted yet
    pos = object_centers(lab)
    calls = pd.DataFrame(dict(id=["1", "2", "3", "4"], call=["correct", "false detection", "fix outline", "fix outline"]))
    missed = pd.DataFrame(dict(z=[2], y=[300], x=[100]))
    ed = tmp_path / "edits"
    ed.mkdir()
    box = lab[1:4, 480:530, 480:530].copy()
    box[1, 5:45, 5:45] = 3                     # painted: object 3 made bigger
    np.savez_compressed(ed / "3.npz", lab=box, lo=np.array([1, 480, 480]))
    n = export_patches(img, lab, pos, calls, missed, tmp_path / "out", size=256, window=(1000, 3000),
                       edits=load_edits(ed))
    assert n == {"ready": 3, "needs_editing": 2}
    root = tmp_path / "out"
    m1 = tifffile.imread(root / "ready" / "review" / "masks" / "review_1.tif")
    assert set(np.unique(m1)) == {0, 1}        # false detection removed from the mask
    m3 = tifffile.imread(root / "ready" / "review" / "masks" / "review_3.tif")
    assert (m3 == 3).sum() == 40 * 40          # painted outline used
    i1 = tifffile.imread(root / "ready" / "review" / "images" / "review_1.tif")
    assert i1.dtype == np.uint8 and i1.shape == (256, 256)
    assert (root / "needs_editing" / "review" / "images" / "review_4.tif").exists()
    assert (root / "needs_editing" / "review" / "images" / "review_missed1.tif").exists()
