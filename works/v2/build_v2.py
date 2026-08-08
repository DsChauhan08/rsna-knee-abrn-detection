#!/usr/bin/env python3
"""Assemble v2 (adaptive MedSLip + Gemma report cache) notebook.

v2 over v1 (fully offline, no downloads, scoreable from second 1):
  * vision: adaptive encoder behind one encode() interface:
      - HF transformers SiglipVisionModel dir (v1 layout)
      - KerasHub preset dir (config.json + model.weights.h5)
      - learned stub (random init, trainable) as last resort
    Every branch is smoke-tested with a zero-image forward at load time; the
    first branch that passes is kept. load_vision() never returns None.
  * text:  Gemma-4E2B report embeddings as a one-shot cache per split:
      attention-mask mean-pooled last hidden state, L2 normalised float16,
      npz keyed by StudyInstanceUID (text_cache_<tag>.npz). GPU only; on CPU
      the v1 extractor text features remain the report branch.
  * model: V2Model = vision.encode() -> (B,S,Dv); report text broadcast per
      slot; concat -> fusion(Linear/LN/GELU/Dropout) -> SlotHead (v1 class,
      VERBATIM). rank loss + pixel-space aug: v1 VERBATIM.
  * EMA: shadow-param EMA on trainable weights only (no deepcopy of a
      possibly-Keras backbone); target(model) loads shadow into a fresh eval
      copy (build_eval_model) so live training state is untouched.

Cell layout mirrors v1; every v1 cell that stays is byte-identical.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent            # works/v2
V1 = ROOT.parent / "v1/rsna_knee_v1_final.ipynb"
OUT = ROOT / "rsna_knee_v2_medgemma.ipynb"
CELLS = ROOT / "cells"

MD_TITLE = """# RSNA Knee Abnormality Detection — v2 (adaptive MedSLipMedSigLIP + Gemma report cache)

Fully offline; top-to-bottom like the v1 kernel; the 0.5 benchmark submission
is written before any model runs.

1. **vision**: adaptive encoder behind one `encode()` interface — HF
   `SiglipVisionModel` dir, KerasHub preset dir (config.json + weights.h5),
   or a learned stub. Each branch is smoke-tested with a zero-image forward;
   the first that passes is kept.
2. **text**: Gemma report embeddings (mean-pooled last hidden state,
   L2-normalised) computed once per split and cached next to the vision
   cache; broadcast per slot and concatenated to the vision features.
3. **head**: v1's SlotHead untouched; EMA is a shadow of trainable weights
   (never deepcopies a possibly-Keras backbone); the averaged snapshot is
   loaded into a fresh copy for validation and submission."""

MD_STEP3 = """## Step 3 — vision: adaptive encoder (HF | KerasHub | stub) + Gemma report cache

`load_vision()` walks every attached candidate (config.json dirs, shortest
path first). MedGELIP-like layouts pass if they have vision weights; the
branch survives only if a zero-image forward returns the expected width.
No encoder attached? The run still trains: a small learned stub features the
raw pixels (worse, but valid and testable on a laptop).

The Gemma cell embeds every report once per split (train/test only; fold
training reuses the same cache). A GPU-less kernel skips Gemma and the v1
extractor text features remain the report branch."""

MD_STEP4 = """## Step 4 — slot-attention head (v1 verbatim), shadow EMA, CV, submission

The head is unchanged from v1: per-diagnosis attention over the six slot
views. `V2Model` adds the fusion (vision + optional text) before the head;
`Ema` tracks shadow averages of the trainable weights and `build_eval_model`
copies them into a fresh eval model — never mutating the training instance.
Augmentation stays pixel-space and runs BEFORE the encoder, exactly like v1.
CV folds, rank loss and the multi-fold rank-averaged submission are v1."""


def code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def markdown(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def join(c):
    return "".join(c["source"])


def pick(cells, needle):
    """Pop the first cell whose joined source contains `needle`."""
    for i, c in enumerate(cells):
        if needle in join(c):
            return cells.pop(i)
    raise KeyError(needle)


nb = json.loads(V1.read_text())
c = nb["cells"]

imports = pick(c, "import os, sys, re,")
targets = pick(c, "# ---- TARGETS")
lexicon = pick(c, "STEM_MENISCUS")
cfg = pick(c, "# ---- configuration")
rootc = pick(c, "# ---- competition root")
labels_c = pick(c, "# ---- labels: inline extractor")
hdr = pick(c, "HDR_TAGS = [")
pick_slots = pick(c, "def pick_slots")
order = pick(c, "ORDER_TAGS = [")
norm = pick(c, "def normalise_laterality")
cache = pick(c, "def build_cache")
# drop the v1 vision-slothead-predict-main bodies (replaced):
for needle in ("def find_medsiglip", "class SlotHead", "def augment",
               "def predict", "# ---- orchestrator"):
    c[:] = [x for x in c if needle not in join(x)]
# old markdown cells are replaced by the new MD_* above
c[:] = [x for x in c if x["cell_type"] != "markdown"]
leftover = [x for x in c if "".join(x["source"]).strip()]
if leftover:
    raise SystemExit(f"unrecognised v1 cells left: {[join(x)[:50] for x in leftover]}")

cells = [
    markdown(MD_TITLE),
    imports,
    markdown("## Step 1 — labels (inline multilingual rule extractor, fully offline)"),
    targets,
    lexicon,
    cfg,
    rootc,
    labels_c,
    markdown("## Step 2 — DICOM → slots → cache"),
    hdr,
    pick_slots,
    order,
    norm,
    cache,
    markdown(MD_STEP3),
    code((CELLS / "vision.py").read_text()),
    code((CELLS / "text.py").read_text()),
    code((CELLS / "model.py").read_text()),
    markdown(MD_STEP4),
    code((CELLS / "augment.py").read_text()),
    code((CELLS / "predict.py").read_text()),
    code((CELLS / "main.py").read_text()),
]

nb["cells"] = cells
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {OUT} ({len(cells)} cells)")