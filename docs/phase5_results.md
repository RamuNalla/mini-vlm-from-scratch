# Phase 5 results (in progress)

Experiments on the Mini-VLM: CLIP ViT-B/16 encoder, learned projector, Qwen2.5-0.5B-Instruct with LoRA.
This file covers what is finished so far: the visual-token count, the projector type, the encoder
(frozen against partly trained) and the training stages. The full-length run, the latency timing and the
plots are still to do.

## Protocol

Every variant uses the same reduced protocol so the runs are comparable and affordable on a local M4 Pro (MPS):

- Stage 1 (align the projector): 1 epoch on 10k COCO captions, 612 steps.
- Stage 2 (projector + LoRA): 1 epoch on 10k VQAv2 records, 625 steps.
- About 50 minutes per variant. One seed per variant.
- Evaluation: 3,000 held-out VQAv2 questions (official soft accuracy, normalised answers) and 1,000 caption images.

**Noise floor.** The standard error on 3,000 questions is about 0.9 points, and each variant is a single seed.
Treat gaps under about 3 points as noise.

**Reference for scale.** The full recipe (3 epochs of stage 1, 2 epochs on 20k VQA records) reaches 57.36% VQA and
caption CIDEr 0.272. Always answering "yes" scores 24.7% and the text-only model scores 24.4%.

## Results

| Variant | Projector | Visual tokens | VQA | yes/no | number | other | CIDEr | ms/question | Trainable (M) |
|---|---|---|---|---|---|---|---|---|---|
| ref | linear | 64 | 52.18 | 74.98 | 36.92 | 39.74 | 0.265 | 40 | 9.49 |
| tokens_16 | linear | 16 | 49.39 | 73.58 | 33.82 | 36.04 | 0.238 | 28 | 9.49 |
| proj_mlp | MLP | 64 | 50.77 | 75.50 | 36.15 | 36.79 | 0.208 | 42 | 10.29 |
| proj_resampler | resampler | 64 | 41.69 | 68.88 | 29.64 | 25.28 | 0.058 | 43 | 16.78 |

Trainable parameters include the 8.8M LoRA weights.

## Visual-token count (16 vs 64)

Going from 64 to 16 tokens costs 2.8 points of VQA accuracy, about the size of the noise floor. The drop is
spread across the categories: number -3.1, other -3.7, yes/no -1.4. Training was twice as fast per step
(about 1.0 s against 2.1 s). The curve has two points only: the 196-token run was stopped as too slow and the
256-token run was skipped, both by decision.

Reading: 16 tokens keep most of the accuracy at half the training cost, but the evidence for any loss is weak.

## Projector type (64 tokens)

- **MLP vs linear:** 1.4 points lower VQA, inside the noise. The MLP's CIDEr is lower (0.208 against 0.265).
  No evidence that the extra layer helps in this budget.
- **Resampler vs linear:** 10.5 points lower, well outside the noise, and CIDEr is 0.058. The resampler has 64
  learned queries that attend to all 196 patches, and about 8M parameters against 0.69M for the linear
  projector. The near-zero CIDEr suggests it had not finished aligning with the language model after 612
  stage-1 steps. The stage-1 learning rate (1e-3) was not tuned, and the stage-1 loss curve was not inspected.
  This is not evidence that resamplers are worse in general, only that this one is under-trained here. It was
  not re-run, by decision.

Reading: under the reduced protocol the simple linear projector is as good as or better than the alternatives.

## Encoder: frozen against last 2 blocks trained

A smaller protocol than the main table: stage 2 only, 3,000 VQA records (187 steps), both runs starting from the
reference run's stage 1 projector. The two runs are compared with each other, not with the 10k-record reference.
The unfrozen run trains the last 2 CLIP blocks that feed the VLM (14.2M extra parameters, learning rate 1e-5)
with images run through the encoder live.

| Variant | VQA | yes/no | number | other | CIDEr | Trainable (M) |
|---|---|---|---|---|---|---|
| enc_frozen_small (frozen CLIP) | 48.13 | 72.15 | 31.26 | 35.23 | 0.272 | 9.49 |
| enc_unfreeze2_small (last 2 blocks) | 48.20 | 72.57 | 29.90 | 35.43 | 0.257 | 23.66 |

Reading: a 0.07 point difference, far inside the noise. Training the encoder gave no benefit in this budget, and
it costs 14M more trainable parameters and a live encoder pass. One seed, 187 steps and an untuned encoder
learning rate: this does not show that fine-tuning can never help. The CIFAR-trained encoder comparison was
skipped by decision.

## Training stages

Same 3,000-record stage 2 as above where a stage 2 is involved.

| Training | VQA | yes/no | number | other | CIDEr |
|---|---|---|---|---|---|
| Stage 1 only (captions, 612 steps) | 15.13 | 2.31 | 24.08 | 22.03 | 0.871 |
| Stage 2 only (random-init projector) | 38.04 | 67.01 | 25.51 | 20.48 | 0.025 |
| Stage 1 + stage 2 (3k records) | 48.13 | 72.15 | 31.26 | 35.23 | 0.272 |
| Stage 1 + stage 2 (10k records, ref) | 52.18 | 74.98 | 36.92 | 39.74 | 0.265 |

For scale: always answering "yes" scores 24.7% and the text-only model 24.4%.

- **Stage 1 alone** describes images well (CIDEr 0.871) but cannot answer questions: it writes captions, so yes/no
  accuracy is 2.3%.
- **Skipping stage 1** costs 10.1 points of VQA accuracy against the aligned start (38.04 against 48.13), outside
  the noise. The largest loss is on "other" questions (20.5 against 35.2). It still beats the text-only model by
  about 14 points, so the model does learn to use the image from VQA data alone. Its caption CIDEr is near zero
  (0.025) because it never saw a caption; the VQA stage also shortens captions in the aligned model (0.871 after
  stage 1, 0.272 after stage 2).
- **Caveat:** the stage 2 projector learning rate (3e-4) was set for an already aligned projector, so it may be
  low for one starting from random weights. Part of the gap could come from that rather than from missing
  alignment itself.

## Caveats

- One seed per variant, 3,000 questions, so small gaps are not reliable.
- Per-question latency from a single run is noisy (the same model has measured 26 ms and 40 ms). Latency against
  token count should be timed separately with repeats before it is quoted.
- Comparisons hold only within the reduced protocol. The best configuration still needs a full-length run to
  confirm it.

## Still to do

1. Full-length run of the best configuration (about 2.7 hours; to be confirmed first).
2. Latency against token count, timed with repeats, and the plots.
