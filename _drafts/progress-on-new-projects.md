---
title: "Progress on New Projects I Am Working On"
date: 2026-11-09
summary: "Two projects in progress, and why I think they matter: finding which retrieved sources an answer actually used, in one forward pass instead of hundreds, and making diffusion language models generate faster without changing what they write."
tags: [Diffusion LMs, Attribution]
---

These are two projects I am in the middle of. The methods are still moving and every number here is
preliminary, so this post is not a results write-up. It is about why I think these two
problems are worth working on.

## Which sources did the answer actually use?

Retrieval-augmented generation is how most deployed language models answer questions about documents
they were never trained on: search, paste the hits into the prompt, answer. Whenever the answer
matters, the next question is which of those retrieved passages it actually relied on. A user deciding
whether to trust a citation needs that, and so does anyone auditing a system that cites sources.

The standard way to find out is credit assignment. Remove passages, re-run the model, see what
changes; the Shapley value is the principled version of that experiment. It has two problems.

- **It is expensive.** Hundreds of extra forward passes per answer (about 420 in our setting) make it
  an offline audit tool, not something you can run on every response.
- **It asks a subtly wrong question.** It measures how likely the model's own answer stays when
  passages are removed, which rewards whatever supports that answer, right or wrong. When a look-alike
  passage misleads the model, Shapley credits the passage that misled it and scores the evidence the
  model should have used as harmful.

The project treats attribution as **compression** instead. Find the smallest part of the context
under which the model still believes what it believed with the whole context. A small selector reads
that off the forward pass that produced the answer, so attribution costs nothing extra at inference.
Trained to preserve the model's full next-token beliefs rather than the likelihood of one answer, it
matches the Shapley value of that belief-preserving game in a single pass, and it keeps more of the
real evidence.

It is useful beyond explanation, too. Once you know which sentences mattered, you can drop the rest
and ask again. On HotpotQA, that second attempt already fixes about a third of the wrong answers.

## Making diffusion language models fast without changing their answers

Diffusion language models such as LLaDA and NVIDIA's Nemotron-Labs-Diffusion can write many tokens per
forward pass instead of one. Generation cost and latency are mostly a count of forward passes, so this
is a direct route to cheaper, faster text. In practice, though, samplers leave most of that parallelism
unused: tokens decoded together cannot see each other, and they can contradict one another.

The idea running through this work is **trajectory tracking**. Pick a decoder you trust but that is
slow, one token at a time, and learn when committing several tokens at once would change what it
produces. That turns speed into a supervised problem whose labels are free: replay the slow decoder
once and you know, at every position, whether a parallel commit would have agreed with it.
[PIG]({{ '/blog/pig-parallel-inconsistency-guidance/' | relative_url }}) was the first version.
Re-Cal predicts that disagreement directly, then fine-tunes the model along the path its own fast
decoder takes, which raises the ceiling on how parallel the model can be.

Nemotron-Labs-Diffusion already has an exact form of trajectory tracking built in: self-speculation.
Its diffusion mode drafts a block of tokens and its autoregressive mode verifies them, so the output is
exactly what the slow decoder would write. But it keeps only the prefix up to the first mismatch.
Everything drafted after that point is thrown away, even when most of it was right. Two directions
follow from that:

- **Waste less of each verification.** Prepare only the drafts that are likely to be needed: a small
  adapter predicts where verification will stop.
- **Stop throwing the tail away.** An edit mode, running inside the verification pass, keeps what is
  right and regenerates only what is not. Editing comes down to two steps: accept any subset of the
  draft, then re-mask the rest and run the model again. That one mechanism covers prefix acceptance,
  acceptance past the first mismatch, and repair, while the autoregressive check keeps the output
  exact.

The goal is the same text in fewer forward passes, with a training signal that costs nothing but
prompts.

Both projects are moving quickly, and the details will change as the experiments land. If you work on
attribution or on fast decoding for diffusion language models, I would be glad to talk.
