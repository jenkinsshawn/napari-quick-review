"""Quick review: step through a segmentation model's objects in a 3D image, make one call per
object with a single key press, fix outlines by painting, and mark objects the model missed.

The image is never loaded whole. For each object only a small box is read from the open
image layer (numpy, zarr, dask; 8 or 16 bit), so the window stays fast on large volumes.
"""
import time

import numpy as np
import pandas as pd
from magicgui import widgets

from ._core import Calls, export_patches, load_edits, object_centers, read_crop, stratified_sample

HELP = ("1 correct   2 fix outline   3 false detection   click: missed object\n"
        "Left back   Up/Down slice   - / = zoom")


class QuickReview(widgets.Container):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.crop_layers = []     # layers made by the reviewer (kept out of the layer lists)

        # settings shown in the side panel
        self.image = widgets.ComboBox(label="image layer", choices=self._layers("Image"))
        self.labels = widgets.ComboBox(label="model output layer", choices=self._layers("Labels"))
        self.output = widgets.FileEdit(label="save calls to", mode="w", filter="*.csv")
        self.number = widgets.SpinBox(label="objects to review (0 = all)", value=100, min=0, max=100000)
        self.half_xy = widgets.SpinBox(label="half width (px)", value=320, min=16, max=4096)
        self.half_z = widgets.SpinBox(label="half depth (slices)", value=30, min=0, max=1024)
        self.paint_on = widgets.CheckBox(text="paint outlines when 2 is pressed", value=False)
        self.start = widgets.PushButton(text="Start")
        self.export = widgets.PushButton(text="Export training patches")
        self.status = widgets.Label(value=HELP)
        self.extend([self.image, self.labels, self.output, self.number,
                     self.half_xy, self.half_z, self.paint_on, self.start, self.export, self.status])
        self.start.changed.connect(self._start)
        self.export.changed.connect(self._export)

        # keep the layer lists up to date as layers are added or removed
        for ev in (napari_viewer.layers.events.inserted, napari_viewer.layers.events.removed):
            ev.connect(lambda e: self.reset_choices())

        self.i = 0                # index of the object on screen
        self.last_press = 0.0
        self.editing = False

    # ---------------------------------------------------------------- setup
    def _layers(self, kind):
        def choices(widget=None):
            return [L.name for L in self.viewer.layers if type(L).__name__ == kind and L not in self.crop_layers]
        return choices

    def _start(self):
        src = self.viewer.layers[self.image.value]
        self.src = src.data[0] if src.multiscale else src.data   # full resolution
        if self.src.ndim != 3:
            raise ValueError("The image must be 3D (z, y, x).")
        self.scale, self.offset = np.array(src.scale), np.array(src.translate)
        L = self.viewer.layers[self.labels.value]
        self.lab = L.data[0] if L.multiscale else L.data
        self.calls = Calls(self.output.value)
        stem = self.calls.path.with_suffix("")
        self.missed_path = stem.with_name(stem.name + "_missed.csv")
        self.edit_dir = stem.with_name(stem.name + "_edits")
        list_path = stem.with_name(stem.name + "_review_list.csv")

        # the objects to review are chosen once and kept in <calls>_review_list.csv, so a review
        # can be stopped and resumed with the same objects
        if list_path.exists():
            self.pos = pd.read_csv(list_path, dtype={"id": str})
        else:
            pos = object_centers(self.lab)            # one stop per object in the model output
            self.pos = stratified_sample(pos, self.number.value, self.src.shape)
            self.pos.to_csv(list_path, index=False)

        # hide the full image so napari does not keep drawing it, and show the crop layers instead
        for L in self.viewer.layers:
            L.visible = False
        self._make_crop_layers()
        self._bind_keys()
        done = self.calls.done()
        self.i = next((j for j, id_ in enumerate(self.pos.id) if id_ not in done), 0)
        self.zoom = None
        self._show()

    def _make_crop_layers(self):
        for L in self.crop_layers:
            if L in self.viewer.layers:
                self.viewer.layers.remove(L)
        self.crop = self.viewer.add_image(np.zeros((1, 2, 2), self.src.dtype), name="review", scale=self.scale)
        self.crop_lab = self.viewer.add_labels(np.zeros((1, 2, 2), np.int32), name="model outlines",
                                               scale=self.scale, opacity=0.8)
        self.crop_lab.contour = 1              # outlines only, so the image underneath stays visible
        # red ring at the object; a 2D shape is drawn on every slice
        self.ring = self.viewer.add_shapes(name="object", ndim=2, scale=self.scale[1:])

        # objects the model missed: click on them; each click is saved at once to <calls>_missed.csv
        old = pd.read_csv(self.missed_path) if self.missed_path.exists() else pd.DataFrame(columns=["z", "y", "x"])
        self.missed = self.viewer.add_points(old[["z", "y", "x"]].to_numpy(float), ndim=3, name="missed (click)",
                                             scale=self.scale, translate=self.offset, size=8, face_color="magenta")
        self.missed.events.data.connect(self._save_missed)
        self.crop_layers = [self.crop, self.crop_lab, self.ring, self.missed]
        self._click_mode()

    def _click_mode(self):
        """Normal state: a click adds a missed object; the outlines are display only."""
        self.crop_lab.contour = 1
        self.crop_lab.mode = "pan_zoom"
        self.missed.mode = "add"
        self.viewer.layers.selection.active = self.missed

    def _bind_keys(self):
        # napari layers have their own keys (for example 1-6 on labels layers) that win over the
        # viewer's while the layer is selected, so the same keys are bound on every review layer too
        keys = {"1": lambda: self._decide("1"), "2": self._paint, "3": lambda: self._decide("3"),
                "Enter": self._save_paint, "Escape": self._cancel_paint, "Left": self._back,
                "Up": lambda: self._step(1), "Down": lambda: self._step(-1),
                "PageUp": lambda: self._step(5), "PageDown": lambda: self._step(-5),
                "-": lambda: self._zoom(1 / 1.5), "=": lambda: self._zoom(1.5)}
        for t in [self.viewer] + self.crop_layers:
            for k, f in keys.items():
                t.bind_key(k, lambda *_, f=f: f(), overwrite=True)

    # ---------------------------------------------------------------- one object
    def _show(self):
        row = self.pos.iloc[self.i]
        center = np.array([row.z, row.y, row.x], float)
        half = (self.half_z.value, self.half_xy.value, self.half_xy.value)
        crop, self.lo = read_crop(self.src, center, half)

        # place the crop where it sits in the full image, so coordinates match the original
        world0 = self.offset + self.lo * self.scale
        self.crop.data = crop
        self.crop.translate = world0
        # contrast from this crop (0.5-99.5 percentile): works for 8 and 16 bit alike
        lo_c, hi_c = np.percentile(crop[crop > 0] if (crop > 0).any() else crop, [0.5, 99.5])
        self.crop.contrast_limits = (float(lo_c), float(hi_c) if hi_c > lo_c else float(lo_c) + 1)
        self.crop_lab.data = read_crop(self.lab, center, half)[0].astype(np.int32)
        self.crop_lab.translate = world0
        edit = self.edit_dir / f"{row.id}.npz"          # show a painted outline if there is one
        if edit.exists():
            self.crop_lab.data = np.load(edit)["lab"]

        # ring 1.2 um across if the scale is in nm, otherwise 150 px; given by its four corners
        r = 600 / self.scale[-1] if self.scale[-1] > 1 else 75
        cy, cx = center[1:]
        corners = np.array([[cy - r, cx - r], [cy - r, cx + r], [cy + r, cx + r], [cy + r, cx - r]])
        self.ring.data = []
        self.ring.add(corners, shape_type="ellipse", edge_color="red", face_color="transparent", edge_width=3)
        self.ring.translate = self.offset[1:]

        world = self.offset + center * self.scale
        self.viewer.dims.set_point(0, world[0])
        self.viewer.camera.center = tuple(world[1:])
        if self.zoom is None:            # first object: fit the crop to the window
            self.viewer.reset_view()
            self.viewer.camera.center = tuple(world[1:])
            self.zoom = self.viewer.camera.zoom
        self.viewer.camera.zoom = self.zoom
        self._title()

    def _title(self, note=""):
        row = self.pos.iloc[self.i]
        done = len(self.calls.done() & set(self.pos.id))
        prev = self.calls.get(row.id) or "-"
        self.status.value = f"{self.i + 1} of {len(self.pos)}: object {row.id}   saved call: {prev}   " \
                            f"({done} done)\n{note or HELP}"
        self.viewer.title = f"object {row.id}   saved call: {prev}   ({done} of {len(self.pos)} done)"
        # key reminder in the corner of the image
        try:
            self.viewer.text_overlay.visible = True
            self.viewer.text_overlay.position = "top_left"
            self.viewer.text_overlay.font_size = 14
            self.viewer.text_overlay.color = "yellow"
            self.viewer.text_overlay.text = (f"object {self.i + 1} of {len(self.pos)}   saved call: {prev}\n"
                                             "1 correct   2 fix outline   3 false detection   click: missed")
        except AttributeError:            # older napari without a text overlay
            pass

    def _save_missed(self, event=None):
        pts = pd.DataFrame(np.round(self.missed.data).astype(int), columns=["z", "y", "x"])
        tmp = self.missed_path.with_suffix(".tmp")
        pts.to_csv(tmp, index=False)
        tmp.replace(self.missed_path)

    # ---------------------------------------------------------------- keys
    def _ok(self):
        # ignore a second press within 0.4 s, so a held or bounced key is counted once
        now = time.monotonic()
        if now - self.last_press < 0.4:
            return False
        self.last_press = now
        return True

    def _decide(self, k):
        if self.editing or not self._ok():
            return
        self.calls.save(self.pos.id.iloc[self.i], k)
        self._next()

    def _next(self):
        done = self.calls.done()
        ids = list(self.pos.id)
        # next object without a call; after the last one, go back to any that were skipped
        nxt = next((j for j in range(self.i + 1, len(ids)) if ids[j] not in done), None)
        if nxt is None:
            nxt = next((j for j in range(len(ids)) if ids[j] not in done), None)
        if nxt is None:
            self.status.value = f"All {len(ids)} objects have a call. Press Export training patches."
            self.viewer.title = "Review finished"
            return
        self.i = nxt
        self._show()

    def _paint(self):
        """Key 2: record "fix outline". If painting is switched on, first let the outline be repainted."""
        if self.editing or not self._ok():
            return
        if not self.paint_on.value:          # call only; the patch goes to needs_editing on export
            self.calls.save(self.pos.id.iloc[self.i], "2")
            self._next()
            return
        self.editing = True
        row = self.pos.iloc[self.i]
        here = int(self.crop_lab.data[int(row.z) - self.lo[0], int(row.y) - self.lo[1], int(row.x) - self.lo[2]])
        self.crop_lab.selected_label = here or int(self.crop_lab.data.max()) + 1
        self.crop_lab.contour = 0           # filled while painting, so gaps are easy to see
        self.viewer.layers.selection.active = self.crop_lab
        self.crop_lab.mode = "paint"
        self._title("Painting: draw or erase the outline (brush and eraser in the layer controls). "
                    "Enter saves, Escape cancels.")

    def _save_paint(self):
        if not self.editing:
            return
        id_ = self.pos.id.iloc[self.i]
        self.edit_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.edit_dir / f"{id_}.npz", lab=np.asarray(self.crop_lab.data), lo=self.lo)
        self.calls.save(id_, "2")
        self.editing = False
        self._click_mode()
        self._next()

    def _cancel_paint(self):
        if self.editing:
            self.editing = False
            self._click_mode()
            self._show()                    # reload the outline as it was

    def _back(self):
        if not self.editing and self._ok() and self.i > 0:
            self.i -= 1
            self._show()

    def _step(self, n):
        z = self.viewer.dims.current_step[0] + n
        self.viewer.dims.set_current_step(0, int(np.clip(z, 0, self.src.shape[0] - 1)))

    def _zoom(self, f):
        self.zoom *= f
        self.viewer.camera.zoom = self.zoom

    # ---------------------------------------------------------------- export
    def _export(self):
        """Write the calls made so far as empanada training patches, next to the calls file."""
        src = self.viewer.layers[self.image.value]
        window = None if self.src.dtype == np.uint8 else tuple(float(v) for v in src.contrast_limits)
        missed = pd.read_csv(self.missed_path) if self.missed_path.exists() else pd.DataFrame(columns=["z", "y", "x"])
        out = self.calls.path.with_name(self.calls.path.stem + "_training_patches")
        n = export_patches(self.src, self.lab, self.pos, self.calls.table, missed, out, window=window,
                           edits=load_edits(self.edit_dir))
        self.status.value = f"{n['ready']} ready, {n['needs_editing']} to edit -> {out}"
