# I built a tiny vision-language model on my laptop, then tried to break it

*Draft. A sequel to my post on building a Vision Transformer from scratch.*

After I finished my ViT project I kept running into the same question: the model could tell me a picture was a dog, but
could it tell me what the dog was doing? Models like LLaVA do that by gluing an image encoder to a language model. I
wanted to build the smallest honest version of that idea and see which parts actually matter. All of it runs on one
MacBook.

## The recipe

The pieces are not exotic.

- **The eyes:** CLIP ViT-B/16, loaded into the ViT code from my last project. That needed a few changes (CLIP uses a
  different activation, a patch bias and a pre-norm), and I wrote a test that checks my outputs against the Hugging Face
  model. That test paid for itself, because every later bug could be blamed on something else.
- **The brain:** Qwen2.5-0.5B-Instruct.
- **The glue:** a projector that turns each image patch into something shaped like a word embedding. The image becomes a
  run of 64 placeholder tokens in the prompt, and those embeddings are overwritten with the projected patches. The
  language model then reads one sequence and has no idea some of it came from pixels.

Training has two stages. First only the projector learns, on image captions, so the two embedding spaces line up. Then
the projector and some LoRA adapters learn to answer questions from VQAv2.

I kept the data small on purpose: 10,000 images and 20,000 questions. If an experiment takes three hours, you run it
once and learn little. If it takes fifty minutes, you can ask questions.

## The number

On 3,000 held-out questions the model scores 57.4%. A model that always says "yes" gets 24.7%. The same language model
without the image gets 24.4%. So it works, and it clearly uses the picture.

Then I did the part I find more interesting, which was trying to understand what that 57% is made of.

## What did not matter

I ran a handful of variants with a shortened recipe, one seed each. The noise on 3,000 questions is about a point, so I
only trusted gaps bigger than about three.

- **Fancier projectors did not help.** A single linear layer scored 52.2%, a two-layer MLP 50.8%. A resampler in the
  style of Q-Former scored 41.7%, which looks like a loss until you notice that its captions were also bad. I think it
  was simply under-trained in my budget, and I did not re-run it, so I will not claim anything about resamplers.
- **Unfreezing the encoder did nothing.** Training the last two CLIP blocks gave 48.2% against 48.1% for the frozen
  version, with 14 million more trainable parameters.
- **Fewer image tokens cost very little.** Going from 64 tokens to 16 lost about 3 points, which is around the noise, and
  made batched answering about 40% faster. I wanted a 196-token point too, but that run was too slow for my laptop, so
  the curve has two points.

What did matter was the first stage. Skipping the caption alignment dropped accuracy from 48.1% to 38.0%. Part of that may
be a learning rate that I tuned for an already-aligned projector, so I would not read it as exactly ten points. But the
model that skipped alignment could not caption at all afterwards (CIDEr 0.03).

## Trying to break it

I split the questions by type. Everything improved a lot over the no-image model, but not equally:

| | counting | colour | spatial | yes/no | other |
|---|---|---|---|---|---|
| no image | 13.2 | 26.1 | 12.7 | 42.8 | 11.4 |
| Mini-VLM | 43.8 | 53.6 | 41.3 | 79.4 | 44.2 |

Two things stood out.

**Part of the score is just the question.** If you answer every question with the most common training answer for its
question type, you get 66% on yes/no and 34% on counting without looking at anything. So the real contribution of the
image is smaller there than the headline gain suggests. Colour, spatial and open questions are almost out of reach for
that kind of guessing, so the model's score on those is mostly the picture.

**It follows the image, even when the image is wrong.** I re-answered all 3,000 questions with the image swapped for
another question's image, and again with a blank, average image. A wrong image changed 63% of the answers and cost 23
points. A blank one cost 17. A wrong image is worse than no image, which tells me the model really reads the picture
rather than leaning on the question. On counting, a wrong image pushed the score to 19%, well under what the question
alone gets.

Then I read failures. About thirty, by hand, which is a small sample, so take the next part as impressions:

- **Counting errors are mostly too low.** The model undercounted 148 times and overcounted 54 times. It is right about
  two thirds of the time for one object, about a third for three, and almost never above four. Elephants (3 against 6),
  ski poles (1 against 4) and picture frames (2 against 7) were typical: small, thin, hidden or numerous things.
- **Colour sometimes goes to the bigger neighbour.** The boat was "blue" (the water). The red hat was "black" (the shirt).
- **It cannot read.** A train labelled "hunslet" became "broadway". At 224 pixels the text is a smear, and the model
  produces a plausible word.
- **Some of my "failures" were the scorer.** Answering "table" where people wrote "restaurant" is an exact-match miss, not
  a vision error.

I also caught myself being sloppy. My first grouping rule put "how much time is left?" into the spatial questions
because of the word "left", and I only noticed because I was reading the examples. The numbers barely moved when I fixed
it, but it was a good reminder to read the data before trusting a table.

## What I would do next

The obvious things: more training data, a higher image resolution so text is legible, and a second seed for each
experiment. I would also give the resampler a proper chance, with a longer first stage and a tuned learning rate, because
my one data point on it is not a fair test. And I would remove parts of a single image, say the top half, to see which
region the model uses, which I only did at the level of whole images.

The code, the experiments and every number above are in the repository, with a notebook for the failure analysis.
