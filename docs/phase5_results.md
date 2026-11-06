# Phase 5 results (in progress)

Experiments on the Mini-VLM: CLIP ViT-B/16 encoder, learned projector, Qwen2.5-0.5B-Instruct with LoRA.
This file covers what is finished so far: the visual-token count and the projector type. The encoder,
training-stage and full-length runs are still to do.

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

## Caveats

- One seed per variant, 3,000 questions, so small gaps are not reliable.
- Per-question latency from a single run is noisy (the same model has measured 26 ms and 40 ms). Latency against
  token count should be timed separately with repeats before it is quoted.
- Comparisons hold only within the reduced protocol. The best configuration still needs a full-length run to
  confirm it.

## Still to do

1. Encoder: frozen against the last 2 blocks unfrozen (planned as a cached block-9 hidden state with the last
   blocks run live).
2. Training stages: stage 1 only, stage 1 + 2, stage 2 only.
3. Full-length run of the best configuration.
4. Latency against token count, timed with repeats, and the plots.
