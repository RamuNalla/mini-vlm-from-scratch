# Phase 6: failure analysis (in progress)

Where the Mini-VLM fails, using the saved predictions on the 3,000 held-out VQAv2 questions. "Full recipe" is the
57.4% model (3 epochs of stage 1, 2 epochs of stage 2); "reduced" runs are the Phase 5 variants. Tables are
produced by `scripts/analyze_failures.py` and `scripts/analyze_priors.py` (no model runs) into
`results/failure_analysis/`.

## Accuracy by question group

Each question is put in one group by rules on its text (`src/analysis/question_groups.py`), in this order:
counting ("how many"), colour, spatial ("where" or a position phrase such as "on the left", behind, next to),
yes/no (the rest with yes/no answers), other. The rules are crude: reading failures showed that an early version
put "how much time is left?" in spatial, so "left" and "right" now count only in position phrases. Accuracy in %, plus or minus one standard error.

| System | counting (n=324) | colour (325) | spatial (216) | yes/no (1056) | other (1079) | all |
|---|---|---|---|---|---|---|
| text-only | 13.2 ± 1.7 | 26.1 ± 2.3 | 12.7 ± 2.2 | 42.8 ± 1.4 | 11.4 ± 0.9 | 24.3 |
| full recipe | 43.8 ± 2.6 | 53.6 ± 2.6 | 41.3 ± 3.1 | 79.4 ± 1.1 | 44.2 ± 1.4 | 57.4 |
| ref (reduced) | 41.7 ± 2.6 | 38.6 ± 2.6 | 34.4 ± 2.9 | 74.9 ± 1.2 | 40.7 ± 1.4 | 52.2 |
| 16 tokens | 38.6 ± 2.6 | 38.2 ± 2.6 | 32.7 ± 2.9 | 73.6 ± 1.3 | 35.6 ± 1.4 | 49.4 |
| MLP | 40.0 ± 2.6 | 38.5 ± 2.6 | 30.0 ± 2.8 | 75.4 ± 1.2 | 37.7 ± 1.4 | 50.8 |

- Every group gains 29 to 37 points over text-only with the full recipe: the model does use the image.
- Colour is where the reduced runs lose the most against the full recipe (38.6 against 53.6, 15 points); the other
  groups lose 2 to 8 points. Colour appears to need the longer training the most.
- Spatial (41.3%) and counting (43.8%) are the weakest groups, far below yes/no (79.4%).
- The 16-token and MLP runs differ from the reference by 2 to 5 points per group, within the noise per group.

## How much is just the question?

A text-only model still scores 42.8% on yes/no from wording alone. A stronger no-image baseline answers every
question with the most common training answer for its question type ("how many" gets "2", "is the" gets "yes").

| Group | model | text-only | question-type prior | model answer equals prior answer | model accuracy when equal / different |
|---|---|---|---|---|---|
| counting | 43.8 | 13.2 | 34.4 | 31% | 52.4 / 40.0 |
| colour | 53.6 | 26.1 | 24.3 | 26% | 57.3 / 52.3 |
| spatial | 41.3 | 12.7 | 9.7 | 9% | 70.0 / 38.5 |
| yes/no | 79.4 | 42.8 | 66.4 | 54% | 81.2 / 77.3 |
| other | 44.2 | 11.4 | 5.6 | 7% | 59.5 / 43.0 |

- **Counting and yes/no lean on the prior the most.** A guess from the question type alone already scores 34.4% on
  counting and 66.4% on yes/no, so the image adds about 9 and 13 points there. Colour, spatial and other
  questions are mostly out of the prior's reach (24, 10 and 6%), so the model's accuracy there comes from the image.
- Yes/no answers are balanced (564 "yes" and 492 "no" predicted, against 560 and 496 in the truth), so the
  model is not just answering "yes".

## Counting

The model answers with 12 distinct values and under-counts far more often than it over-counts:

- 148 under-counts, 54 over-counts and 119 exact answers out of 321 numeric answers.
- Exact accuracy falls with the true count: 53% at 0, 64% at 1, 49% at 2, 34% at 3, 10% at 4, 12% at 5 and 0% at 6.
- It predicts "1" 109 times (truth: 64) and "2" 98 times (truth: 82), so it leans toward small numbers.

One possible reason is that pooling the 196 patches down to 64 tokens blurs small objects, so crowded scenes are
hard to count. That is a guess, not something tested yet (the 16-token run, which pools harder, scores 38.6 on
counting against 41.7 for 64 tokens, a gap inside the noise). The examples in step 3 should show more.

## What the failures look like

Read by hand from 8 random failures (soft accuracy 0) in each of four groups, 32 in total
(`python -m scripts.failure_examples`, seeded; contact sheets in `results/failure_analysis/examples_*.png`). This
is a small, subjective sample, so the counts below show what is common, not how common.

**Counting (7 of 8 under-counts).** Elephants 3 against 6, empty seats 1 against 4, ski poles 1 against 4, boxes on
the top row 1 against 5, picture frames 2 against 7, signs 2 against 6, birds 4 against 7. The objects are small, thin,
partly hidden or numerous. The remaining one is "how many colors is the tennis ball", a colour question that the
"how many" rule counts as counting.

**Colour (5 of 8 take the colour of a larger nearby thing).** The boat is "blue" (the water), the stripes in the grass
are "green" (the grass), the boy's red hat is "black" (his shirt), the surfboard is "red" and the sunglasses
"black" when they are white and pink. Two more are near shades or multi-colour objects: walls "brown" against
"beige", a brown-and-black cat called "orange". One is text colour on a bus ("blue" against "red").

**Spatial (4 of 8 are plausible but not matching).** "Where is the water" gets "in river" against "on left", "where are
the sitting" gets "table" against "restaurant", "sheep standing next to" gets "fence" against "barn", and "on the
pavement beside the child" gets "bike" against "shadow". Exact-match scoring punishes answers at a different level of
detail or of the right kind but not the one people wrote. Only 2 of the 8 are clear position errors
(snow against trees at the top of the mountains, "on table" against "under bench" for the empty cups).

**Other (4 of 8 need reading text).** "What letters on the front of the train" gets "s" against "mata", "what does the
front of the train say" gets "broadway" against "hunslet" (a made-up but plausible word), "what number is in the white
circle" gets "1" against "250", and the cart seller sells "food" against "veggie burgers" (written on the sign). The
others are fine details or small objects: a donut called "hot dog", a triangle called "diamond", a dog nobody can see
at that size, and "what type of breakfast" answered with "breakfast" (echoing the question).

**Summary.** Failures come mostly from (1) small, numerous or hidden objects at 64 tokens, (2) colour taken from a
bigger neighbour, (3) text that cannot be read at 224 px, and (4) strict answer matching. None of this is
tested yet: the next step uses an image ablation on a few examples to see what the model really uses.

## Still to do in Phase 6

The image-token ablation on a few examples, the notebook `notebooks/03_failure_analysis.ipynb`, the final README
and the blog draft.
