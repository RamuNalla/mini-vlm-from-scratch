# Phase 6: failure analysis (in progress)

Where the Mini-VLM fails, using the saved predictions on the 3,000 held-out VQAv2 questions. "Full recipe" is the
57.4% model (3 epochs of stage 1, 2 epochs of stage 2); "reduced" runs are the Phase 5 variants. Tables are
produced by `scripts/analyze_failures.py` and `scripts/analyze_priors.py` (no model runs) into
`results/failure_analysis/`.

## Accuracy by question group

Each question is put in one group by rules on its text (`src/analysis/question_groups.py`), in this order:
counting ("how many"), colour, spatial ("where" or a position word such as left, behind, next to), yes/no (the
rest with yes/no answers), other. Accuracy in %, plus or minus one standard error.

| System | counting (n=324) | colour (325) | spatial (233) | yes/no (1051) | other (1067) | all |
|---|---|---|---|---|---|---|
| text-only | 13.2 ± 1.7 | 26.1 ± 2.3 | 13.6 ± 2.1 | 42.7 ± 1.5 | 11.5 ± 0.9 | 24.3 |
| full recipe | 43.8 ± 2.6 | 53.6 ± 2.6 | 41.6 ± 3.0 | 79.4 ± 1.1 | 44.3 ± 1.4 | 57.4 |
| ref (reduced) | 41.7 ± 2.6 | 38.6 ± 2.6 | 36.0 ± 2.8 | 75.0 ± 1.2 | 40.6 ± 1.4 | 52.2 |
| 16 tokens | 38.6 ± 2.6 | 38.2 ± 2.6 | 33.7 ± 2.8 | 73.6 ± 1.3 | 35.6 ± 1.4 | 49.4 |
| MLP | 40.0 ± 2.6 | 38.5 ± 2.6 | 32.4 ± 2.8 | 75.3 ± 1.2 | 37.6 ± 1.4 | 50.8 |

- Every group gains 28 to 37 points over text-only with the full recipe: the model does use the image.
- Colour is where the reduced runs lose the most against the full recipe (38.6 against 53.6, 15 points); the other
  groups lose 2 to 8 points. Colour appears to need the longer training the most.
- Spatial (41.6%) and counting (43.8%) are the weakest groups, far below yes/no (79.4%).
- The 16-token and MLP runs differ from the reference by 2 to 5 points per group, within the noise per group.

## How much is just the question?

A text-only model still scores 42.7% on yes/no from wording alone. A stronger no-image baseline answers every
question with the most common training answer for its question type ("how many" gets "2", "is the" gets "yes").

| Group | model | text-only | question-type prior | model answer equals prior answer | model accuracy when equal / different |
|---|---|---|---|---|---|
| counting | 43.8 | 13.2 | 34.4 | 31% | 52.4 / 40.0 |
| colour | 53.6 | 26.1 | 24.3 | 26% | 57.3 / 52.3 |
| spatial | 41.6 | 13.6 | 10.8 | 9% | 72.9 / 38.5 |
| yes/no | 79.4 | 42.7 | 66.4 | 54% | 81.2 / 77.4 |
| other | 44.3 | 11.5 | 5.6 | 7% | 59.5 / 43.2 |

- **Counting and yes/no lean on the prior the most.** A guess from the question type alone already scores 34.4% on
  counting and 66.4% on yes/no, so the image adds about 9 and 13 points there. Colour, spatial and other
  questions are mostly out of the prior's reach (24, 11 and 6%), so the model's accuracy there comes from the image.
- Yes/no answers are balanced (563 "yes" and 488 "no" predicted, against 557 and 494 in the truth), so the
  model is not just answering "yes".

## Counting

The model answers with 12 distinct values and under-counts far more often than it over-counts:

- 148 under-counts, 54 over-counts and 119 exact answers out of 321 numeric answers.
- Exact accuracy falls with the true count: 53% at 0, 64% at 1, 49% at 2, 34% at 3, 10% at 4, 12% at 5 and 0% at 6.
- It predicts "1" 109 times (truth: 64) and "2" 98 times (truth: 82), so it leans toward small numbers.

With 64 visual tokens each about a quarter of the picture's detail, scenes with many small objects are likely
out of reach. That is an interpretation, not something tested yet (step 4 looks at examples).

## Still to do in Phase 6

Error inspection with examples, an image-token ablation on a few examples, the notebook
`notebooks/03_failure_analysis.ipynb`, the final README and the blog draft.
