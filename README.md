# mkt-6121-gui

> An extensible GUI application for real‑time classical **and** AI‑powered image processing, developed for the MKT‑6121 *Machine Vision Applications* course.


##  Preview


## Features

* **Collapsible parameter groups** to keep the interface tidy.
* Live **GPU‑accelerated preview** with threaded processing.
* Classical ops: blur, edge detection, morphology, color space conversions.
* Geometric ops: resize, rotate around arbitrary pivot, perspective warp.
* AI ops: YOLOv5 object detection, U‑Net segmentation, anomalib, pix2pix (plug‑and‑play models).

##  Installation

#  Clone the repo
$ git clone [https://github.com/penyepesto/mkt-6121-gui/tree/main].git
$ cd mkt-6121-gui

#  (Recommended) create a virtual env
$ python -m venv .venv && source .venv/bin/activate 

# Install dependencies
$ pip install -r requirements.txt

## Usage

```bash
python -m gui.run_app
```
