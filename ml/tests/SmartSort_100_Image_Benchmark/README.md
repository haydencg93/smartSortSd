# UPDATED: Automatic bin-label generation

`prepare_neoavara.py` now fills `expected_bin`, `reviewed=auto`, `reviewer`, `rule_source` and `notes` automatically. It uses material-level routing for a standard Iowa City recycling bin: cardboard/metal/paper/plastic = recycling; glass/trash = trash. **These are provisional assumptions, not verified item-by-item ground truth** (especially plastic subtype, cleanliness and material condition). The HTML report displays an explicit warning and `summary.json` records the count of auto-labeled images. Do not present the combined result as verified Iowa City disposal accuracy until independently reviewing the relevant items. Glass bottles/jars can be recycled at separate drop-off locations. Existing manifest files are not overwritten.

If you already prepared the 100 images, you do not need to redownload them: run `python SmartSort_100_Image_Benchmark/autofill_manifest.py --manifest heldout_neoavara/manifest.csv`. Backups are created automatically.

# Smart Sort — 100-image ML-only benchmark

**Existing `app.py`, `train_data.py`, `setup_data.py` and `.pth` files are not modified.**

## Source and labeling warning

- Dataset: [NeoAivara/Waste_Classification_data](https://huggingface.co/datasets/NeoAivara/Waste_Classification_data) (Hugging Face; **Apache 2.0 is claimed by uploader**, but underlying original-image rights and overlap with TrashNet/other sources are not independently established).
- 15,515 total records, 12 original categories. We reserve 100 examples across six material classes (20 cardboard, 20 glass, 20 metal, 20 paper, 10 plastic, 10 trash), using the uploader's class labels.
- This dataset does **not** supply verified Iowa City disposal labels. The generated manifest intentionally leaves `expected_bin` blank and `reviewed=no`.
- **You must visually inspect every image and check the item against current [Iowa City recycling guidance](https://www.icgov.org/government/departments-and-divisions/public-works/resource-management/recycling)**, noting cleanliness, shape, accepted item type, and whether your intended installation is a standard public bin or other collection route. Record `expected_bin` (`recycling`/`trash`), `reviewed=yes`, `reviewer`, `rule_source` (link or local documented rule), and an **item-specific** `notes` justification. If a photograph is ambiguous, replace it with a different image of the same material; do not mark it "Ignored".
- Do not use these 100 images for model training, validation, hyperparameter selection, or confidence tuning. They are a fixed holdout. If any image was used during original pretraining, it is not an independent sample; source provenance must be audited. Hash checks find byte-for-byte overlaps, but cannot prove independence from visually similar images or unknown previous training images.
- `plastic → recycling` in your **existing model's fixed rules** is a simplification. The benchmark keeps that actual prediction for comparison, and **uses separately reviewed Iowa City bin ground truth**; thus incorrect route decisions fail, even if material is correct.

## Install

```bash
python -m pip install torch torchvision pillow matplotlib datasets pytest
```

## Use (from a directory containing your existing checkpoint)

```bash
# 1. Download and select 100 sample images — needs internet and may take a while
python SmartSort_100_Image_Benchmark/prepare_neoavara.py --out heldout_neoavara

# 2. Open heldout_neoavara/manifest.csv. Inspect all 100 images and complete
#    reviewed, reviewer, expected_bin, rule_source, item-specific notes.
#    A missing review is a STOP/validation error, NOT "Ignored".

# 3. Optional, strongly recommended: generate hashes from known original training
#    and new training roots before running. Include ALL accessible train+val images.
python SmartSort_100_Image_Benchmark/hash_training.py data_6class expanded_data --output training_hashes.txt

# 4. Run EXACTLY 100 images on the existing six-class model
python SmartSort_100_Image_Benchmark/benchmark.py \
  --manifest heldout_neoavara/manifest.csv \
  --images heldout_neoavara/images \
  --checkpoint smart_sort_mobilenetv2_6class.pth \
  --training-hashes training_hashes.txt \
  --output results_100

# Test status logic without downloading images
python -m pytest SmartSort_100_Image_Benchmark/tests -v
```

For Windows PowerShell, write the long `benchmark.py` command on one line instead of using `\` continuations. GPU optional: `--device cuda` if PyTorch supports CUDA.

## Generated outputs

- `results_100/SmartSort_100_Image_Report.html` — **offline, self-contained HTML report** with thumbnails, status cards, bar charts of **all 100 timings**, pie chart of status distribution, bar chart of **all 100 confidences**, and searchable/filterable 100-row table with expected/predicted material and bin, latency, confidence and concern/failure explanation.
- `results_100/results.csv` — raw item-level records.
- `results_100/summary.json` — aggregated statistics and thresholds.

## Four color statuses

| Status | Condition | Color |
|---|---|---|
| Passed (Fully) | Correct material AND Iowa City bin; confidence ≥75% AND latency ≤2500 ms | GREEN |
| Passed (w/ Concern) | Both correct; confidence <75% **or** latency >2500 ms | BLUE |
| Failed | Wrong material, wrong bin, or model inference error, regardless of confidence/time | RED |
| Ignored | **Reserved for future functionality only**; never generated by the current 100-image benchmark | YELLOW |

**Latency** starts immediately before opening/decoding an image, includes resize/normalization and model forward pass, and ends after extracting the predicted material/bin. It excludes loading the checkpoint, network transfer, camera capture, physical display and HTML generation. The first image might take longer due to cold initialization. Report clearly labels this as *image processing plus model inference*, not complete physical-device response time.

**No measured 100-image results are included until the real images and item-level ground truth have been reviewed.** The report is generated only when the 100-image test is actually run.
