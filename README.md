# napari-quick-review

A napari plugin for checking the output of a segmentation model one object at a time and turning
the checked objects into training patches for empanada.

The plugin shows a small box of the image around each object, with the model outlines and a red
ring at the object. A key press records the call and moves to the next object. Wrong outlines can
be marked for fixing or repainted in the same window, and objects the model missed can be marked
with a click. Calls
are saved after every key press, so a review can be stopped and continued later.

Only the box around the current object is read from disk. The plugin therefore works on large
volumes (OME-Zarr, dask or memory-mapped arrays) and on 8-bit or 16-bit images.

## Installation

    pip install git+https://github.com/jenkinsshawn/napari-quick-review.git

Or, from a downloaded copy of this folder:

    pip install -e .

The plugin is then listed in napari under Plugins > Quick review.

## Review

1. Run the model on the volume (for example with empanada 3D Inference).
2. Open the image and the model output in napari.
3. In the Quick review panel, select the image layer, the model output layer and a file for the
   calls.
4. Enter the number of objects to review (0 reviews all of them) and press Start.

The objects are sampled across the volume. The volume is divided into 3 x 3 x 3 blocks in depth,
height and width, and objects are drawn from each block in turn, so that crowded regions do not
dominate the sample. The sample is saved to `<calls>_review_list.csv` and reused when a review is
continued. Finding the objects reads the whole model output layer once at the start.

| Key | Call or action |
|---|---|
| 1 | correct (real object, outline fits) |
| 2 | fix outline (real object, outline wrong) |
| 3 | false detection (not the structure) |
| Mouse click | object missed by the model |
| Left arrow | previous object; its call can be changed |
| Up and Down arrows | one slice up or down |
| Page Up and Page Down | five slices up or down |
| - and = | zoom out and in |

By default, key 2 only records the call. If "paint outlines when 2 is pressed" is checked, key 2
switches the outlines to painting instead; Enter saves the repainted outline and Escape cancels.
The brush size and eraser are in the layer controls. Only the slice through the red ring needs to
be repainted, because that slice is used as the training patch. Outlines marked for fixing but not
repainted can be corrected after export (see Training).

A second key press within 0.4 s is ignored, so a key held down is counted once.

## Output files

All files are written to the folder of the calls file.

| File | Contents |
|---|---|
| `<calls>.csv` | one row per object (id, call, key, time); a changed call replaces the earlier row |
| `<calls>_review_list.csv` | objects selected for review |
| `<calls>_missed.csv` | one row per click (z, y, x in image pixels) |
| `<calls>_edits/` | repainted outlines, one file per object |
| `<calls>_training_patches/` | training patches written by Export training patches |

## Training

1. Press Export training patches. Each reviewed object and each click is saved as a 256 x 256
   image and mask of the slice through it, in the folder layout used by empanada
   (`<name>/images/*.tif` and `<name>/masks/*.tif`). The mask is the model output with the
   repainted outlines added and the false detections removed. Patches that can be used directly
   are written to `ready/`. Patches with an outline that was marked for fixing but not repainted,
   or with a missed object, are written to `needs_editing/`; their masks need to be corrected in
   napari before use.
2. In empanada, open Finetune a model (or Train a model) and set the train directory to `ready/`.
   Corrected patches from `needs_editing/` can be moved there as well.
3. Run the new model and review it with a new calls file to check whether it improved.

## Gray scale

empanada trains on 8-bit images. Each training patch is read as an 8-bit grayscale image, so a
16-bit patch would be reduced by dividing its values by 256. For this reason, 16-bit images are
converted to 8 bit on export using the contrast limits of the image layer, which should be set for
the whole volume before exporting. The limits used are saved in `gray_window.txt`. The new model
should be run on an 8-bit copy of the volume made with the same limits.
