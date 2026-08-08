# RSNA Knee ABNR — Session State & Handoff Notes

Last updated: 2026-08-09 (end of session, saved before computer shutdown)

## Goal
Ship a single fully-offline Kaggle notebook (`works/v1/rsna_knee_v1_final.ipynb`) that is
**multimodal**: `keras/medsiglip` vision encoder + a **Gemma 4** text encoder for the
radiology reports + our SlotHead head, to reach leaderboard level (~0.88+ gold; #1 ~0.934).

## Pipeline snapshot (what the notebook does today)

- Source: `rsna-knee-baseline-v1.ipynb` + `rsna-knee-public-4-fold-dinov2-v4.ipynb` (repo root).
- Builder: `/tmp/opencode/build_v1_final.py` -> `works/v1/rsna_knee_v1_final.ipynb` (21 cells, 16 code).
- Config: `IMG=448`, `GROUP=3`, `CACHE_SLICES=3`, `N_SLOT=6`, `CACHE_BUDGET_MAX_GB=24`,
  `ORDER_BUDGET_S=5400`, `TIME_BUDGET=8*3600`; 12 targets; 4 folds; `EMA_DECAY=0.997`;
  `RANK_LOSS_W=0.05`; `GOLD_WEIGHT=3.0`.
- Cell map (code cells): [1]=imports/T0/log, [3]=TARGETS, [4]=extractor, [5]=config,
  [9]=probe/walk/annotate/laterality_maps (v4), [10]=pick_slots, [11]=order_slices/read_slot,
  [12]=normalise_laterality, [13]=build_cache, [15]=find_medsiglip, [16]=SlotHead/FeatModel/build_model,
  [18]=augment/Ema/rank_loss, [19]=predict/macro_auc/write_submission, [20]=main.

## Modal updates (current decision, NOT yet wired in)

- **Text branch = Gemma 4 e4b** (replaces the earlier T5Gemma-2 choice). From `google/gemma-4-e4b-it`,
  loaded via `transformers` (`AutoModelForCausalLM` / `AutoModelForMultimodalLM` + `AutoProcessor`).
  - E4B: text+image+audio, 4.5B effective / 8B params w/ embeddings, 42 layers, 128K ctx, vocab 262K,
    sliding window 512, BF16 ~17.9 GB. Kaggle page: https://www.kaggle.com/models/google/gemma-4
  - Vision encoder alone would be ~150M; but MedSigLIP is the vision encoder decision (next bullet).
  - Heavier than needed for a text embedding; consider only the text backbone / encoder path,
    mean-pooled final hidden state as the "report embedding" fused into SlotHead.
- **Vision encoder = keras/medsiglip** (NOT google HF gated route). Kaggle Models handle:
  - `https://www.kaggle.com/models/keras/medsiglip` -> KerasHub preset `medsiglip_900m_448`,
    900M total (400M vision + 400M text), 448px. Also HF mirror: `keras/medsiglip_900m_448`.
  - Load via `keras_hub.models.Backbone.from_preset("kaggle://keras/medsiglip/...")` or a
    local preset dir under `/kaggle/input/...`. Files layout is a Keras preset dir
    (config.json, model.weights.h5, etc.), NOT a HF transformers checkpoint.
  - This means `find_medsiglip()` must be extended: currently only scans `/kaggle/input` for a
    `config.json` mentioning "medsiglip"; it will not see the Keras preset. The kernel code cell
    must load Keras weights and either (a) port them into a torch `SiglipVisionModel` skeleton, or
    (b) run feature extraction through the Keras model. **This load/port path is the open design
    item for the next session.**

## What the user did (already confirmed by them)

- User has consented to both Kaggle Models and can attach them directly as notebook inputs:
  - `keras/medsiglip` (Keras format)
  - `google/gemma-4` / gemma-4-E4B (transformers format)
- User ran the current notebook on Kaggle GPU (P100) + CPU fallback. Third run log still showed
  `MedSigLIP NOT ATTACHED — using stub encoder`, `compute device: cpu`.

## Environment / platform notes

- Notebook must run fully offline on Kaggle. `kagglehub` + `keras_hub` are available on Kaggle.
- On the local laptop there is **NO** kagglehub / keras_hub (checked: ModuleNotFoundError) so the
  Keras load/port can only be verified on Kaggle or after `pip install`.
- P100 / sm_60 kernels missing in new torch builds -> `pick_device()` CPU fallback already in builder
  (`main()`). Docs updated to recommend T4 x2 or 2024-era container image.
- Very old Kaggle GPU may OOM on a 900M Keras model + training. Consider features-only extraction
  of MedSigLIP for caching.
- Real data: train (4407, 14) studies; cache (4407,6,3,448,448) ~14.8 GB; ordering 20130 slot-series
  / 678385 headers ~1850-2350 s on Kaggle mount; test (3,1) in the user's local copy.

## Completed so far (apply to next session)

- Notebook builder fixed (all 16 code cells parse): `\"` escapes; single-line `find_medsiglip`
  `if`; `nn.LazyLinear` -> `nn.Linear(3*32*32, dim)`; `augment(rows[:,:,:GROUP])` wired in;
  dropped duplicate cell 9; removed `TARGETS` from extractor (strip order + `^.*?TARGETS` regex);
  `unicodedata` import added; extractor self-test removed from builder.
- `pick_device()` cuda->CPU fallback, logged; markdown "How to run" updated.
- E2E harness passes in ~44s CPU: `/tmp/opencode/e2e_gen.py` + e2e_run.py (cwd `/tmp/opencode/e2e`;
  28 train/12 test fake studies, UID prefix 1.2.826.0.1.3680043.9.3674.0); submission.csv columns/UIDs
  match sample_submission.csv. Fold "too small, skipped" is expected on tiny data.

## Next session (immediately after wake-up)

1. Wire `keras/medsiglip` into `find_medsiglip()` + the vision backbone load path; decide
   port-to-torch vs Keras inference. Verify mount layout (Keras preset dir under /kaggle/input).
2. Wire Gemma-4-E4B text encoder for report -> "report embedding", fuse into SlotHead head
   (concatenate or bilinear); keep stub fallback and E2E green.
3. Rebuild from `build_v1_final.py`, re-run `/tmp/opencode/e2e_run.py`, update md run notes.
4. When user back: have found a loadable, container-compatible model set the user can attach.

## Key links / references

- Keras MedSigLIP card: https://huggingface.co/keras/medsiglip_900m_448 (example + preset names)
- KerasHub SigLIPBackbone ./SigLIPVisionEncoder: https://keras.io/keras_hub/api/models/siglip/siglip_backbone/
- Gemma 4: https://www.kaggle.com/models/google/gemma-4 (transformers: AutoModelForCausalLM + AutoProcessor)
- Gemma 4 card: https://ai.google.dev/gemma/docs/core/model_card_4; architecture: transformers README gemma4

## Files touched/located

- Repo: `rsna-knee-baseline-v1.ipynb`, `rsna-knee-public-4-fold-dinov2-v4.ipynb`, `works/v1/rsna_knee_v1_final.ipynb`, this file.
- Builder: `/tmp/opencode/build_v1_final.py` (edit here; rebuild notebook + E2E).
- E2E: `/tmp/opencode/e2e_gen.py`, `/tmp/opencode/e2e_run.py` (cwd `/tmp/opencode/e2e`).
- No git commits yet in repo.