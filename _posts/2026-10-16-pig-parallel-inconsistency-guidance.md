---
title: "PIG: Which Tokens Can a Diffusion Language Model Commit Together?"
date: 2026-10-16
summary: "Parallel decoding is fast until tokens that cannot see each other disagree. PIG predicts, from the forward pass the model already makes, which tokens would come out differently if decoded one at a time, and commits only the ones that would not. A 2M-parameter adapter, under 0.1% of a forward pass, trained once per model: same outputs, fewer passes."
repo: https://github.com/Quentin-Fruytier/PIG-DLM-release
image: /images/pig/hero-card.gif
tags: [Diffusion LMs]
math: true
---

[Paper (arXiv:2610.17375)](https://arxiv.org/abs/2610.17375) &middot;
[Code](https://github.com/Quentin-Fruytier/PIG-DLM-release) &middot;
[LLaDA-8B-Instruct-PIG](https://huggingface.co/UT-IFML/LLaDA-8B-Instruct-PIG) &middot;
[Nemotron-Labs-Diffusion-3B-PIG](https://huggingface.co/UT-IFML/Nemotron-Labs-Diffusion-3B-PIG)

*With Anton Xue, Sanjay Shakkottai and Sujay Sanghavi (UT Austin).*

![Fast-dLLM and PIG.INT decoding the same HumanEval problem on LLaDA-8B-Instruct]({{ '/images/pig/hero-llada-humaneval2.gif' | relative_url }})

*Two decodes of the same prompt by the same model (LLaDA-8B-Instruct, 64-token
blocks). Every frame is one forward pass; a token's colour is the pass that
committed it. Left: Fast-dLLM, which commits every token above a confidence
threshold. Right: PIG.INT, which commits as many tokens as its predicted
inconsistency budget allows. Both produce the identical program. One needs 30
passes, the other 19. Over all 164 HumanEval problems at this geometry the two
operating points sit at 7.2 and 10.1 tokens per forward pass, at 36% and 34%
pass@1 (Wilson 95% intervals of about seven points).*

**At a glance**

- **Same model, same outputs, fewer forward passes.** PIG changes only which
  masked positions get committed together. The backbone is frozen and the
  tokens it produces are the ones it would have produced anyway.
- **Almost free at inference.** The adapter has about 2M parameters and costs
  under 0.1% of the arithmetic of one forward pass; the whole selection step
  adds about 13 ms per round on a shared H100, an upper bound.
- **Cheap to train, trained once.** One harvest of exact finite differences
  (26 GPU hours on a single H100) and about an hour to fit the adapter, on a
  small training-side mixture. The same adapter then serves every benchmark,
  budget and block geometry, including MMLU, which it never saw.
- **Up to 1.7× the tokens per forward pass of Fast-dLLM at matched accuracy
  on LLaDA**, 1.4× over Nemotron's native decoder, and up to 28 more accuracy
  points at the baseline's fastest settings.

## The problem: tokens committed together cannot see each other

A masked diffusion language model predicts every masked position from the same
canvas in a single forward pass. That is where its speed comes from: you can
reveal many positions per pass instead of one. It is also where the trouble
comes from. Positions committed in the same pass never get to condition on each
other. If one slot could be "dog" or "cat" and another "barks" or "meows", each
prediction is fine on its own, and committing them together can still produce
"dog meows".

Decoding one token at a time avoids this, because the canvas is updated after
every commit. It is the slow oracle: it respects the model's own joint
distribution, and it throws away all the parallelism.

Existing parallel decoders pick positions using two kinds of signal. Confidence
methods, with [Fast-dLLM](https://arxiv.org/abs/2505.22618) as the standard
baseline, commit whatever the model is sure about. Dependence methods such as
[DAPD](https://arxiv.org/abs/2603.12996) avoid committing positions that look
coupled, for instance through attention. Both are proxies for the thing we
actually care about.

## A different question

Take a candidate set of positions the decoder is about to commit in parallel,
in some order. Ask: *would the model have written the same tokens if it had
decoded those positions one at a time, in that order?* Any disagreement is what
we call **sequential–parallel inconsistency**. It is not confidence and it is
not dependence; it is the error that parallelism actually introduces, measured
against the sequential trajectory of the same model.

$$
\textbf{sequential:}\;\; \mathbf{x}_t \xrightarrow{x^*_{s_1}} \mathbf{x}_{t+1} \xrightarrow{x^*_{s_2}} \cdots \xrightarrow{x^*_{s_k}} \mathbf{x}_{t+k}
\qquad
\textbf{parallel:}\;\; \mathbf{x}_t \Rightarrow \{\hat{x}_{s_1},\dots,\hat{x}_{s_k}\}
$$

![Overview of PIG: the estimator, the greedy walk, and one round of parallel decoding against the sequential target]({{ '/images/pig/pig_overview.png' | relative_url }})

*(A) A small predictor on the frozen model estimates how committing one
position would change the entropy at every other. (B) Candidates are admitted
in decreasing confidence until the predicted regret passes a budget. (C) One
parallel round against its sequential target: "the cat sat on a mat" versus
"the cat sat on a car" is the inconsistency we want to avoid.*

If this inconsistency were known before committing, parallel decoding would be
a plain selection problem: commit as many positions as you can while the
predicted disagreement stays small. The catch is that measuring it exactly
means replaying the candidates one at a time, which is the slow decode we were
trying to avoid.

## You do not have to replay

It turns out the replay is predictable from the forward pass you already have.
Committing position $$j$$ changes the model's uncertainty at every other
position $$i$$. We write that change as an entropy finite difference,

$$
\mathbf{D}_{ij} = H_i(\mathbf{x}_t) - H_i(\mathbf{x}_t \oplus \hat{x}_j),
$$

signed: positive when committing $$j$$ would make $$i$$ more certain, negative
when it would make $$i$$ less certain (the two predictions are in tension).
The **Finite Differences Estimator** (FDE), a lightweight adapter on the frozen
backbone with about 2M parameters and under 0.1% of a forward pass in
arithmetic, predicts the whole matrix $$\hat{\mathbf{D}}$$ from the hidden
states of one pass. On held-out benchmarks it reaches a correlation of about
0.8 per edge and 0.9 on the aggregate statistics the decoder actually consumes.

![From the canvas to the dependence graph]({{ '/images/pig/m3_pipeline.png' | relative_url }})

*One forward pass of the frozen model gives per-position features (confidence,
entropy, hidden states); the Finite Differences Estimator turns them into
$$\hat{\mathbf{D}}$$ for the whole canvas with one matrix product.*

Here is what the matrix looks like on a real pass of the decode above, for the
24 most confident candidates in the block:

![Predicted entropy finite differences on one pass of the HumanEval decode]({{ '/images/pig/dhat-heatmap.png' | relative_url }})

*Rows are positions whose prediction would change, columns are the candidates
that could be committed; blue means committing the column would make the row
more certain, red that it would unsettle it. The dashed lines separate what
PIG.INT committed on this pass from what it left for the next one. Notice how
little the six committed tokens disturb each other, and how much the waiting
ones would.*

## From a matrix to a decision: spend a regret budget

A matrix is not a decision. PIG turns it into one with a calibration table
$$\hat p(\text{confidence}, \hat\mu)$$: the empirical probability that a token
committed in parallel differs from its sequential reference, given its own
confidence and the aggregate influence $$\hat\mu$$ of the more confident tokens
admitted before it. The table is fitted once per backbone on training-side
data and never touched again.

**PIG.INT** then walks the candidates in decreasing confidence and keeps
admitting them until the predicted number of mismatches in the round exceeds a
budget $$\varphi$$. That is the whole rule. One knob, one direction: larger
$$\varphi$$ is faster, smaller is closer to sequential decoding.

![PIG.INT controls single-round regret]({{ '/images/pig/arxiv_pigint_walk.png' | relative_url }})

*The walk, as drawn in the paper: one forward pass ranks the masked slots by
confidence (I); candidates are added with their predicted incremental regret
(II) and admitted while the total fits the budget $$\varphi$$ (III); the first
candidate that does not fit closes the round.*

The same walk on a real pass of the decode above:

![PIG.INT budget walk on one forward pass]({{ '/images/pig/budget-walk.gif' | relative_url }})

*Each admitted token adds its predicted mismatch
probability to the budget; the first token that would push the total past
$$\varphi$$ closes the round, and everything after it waits for a fresh
prediction conditioned on what was just committed.*

Note what the walk is *not* doing. It never asks whether two tokens are
dependent in the abstract. A pair can be strongly coupled and still both be
admitted, if the coupling is the agreeing kind; a weakly coupled token can be
refused, if it sits behind many admitted tokens that each nudge it a little.

## PIG.EXT: is waiting worth a pass?

PIG.INT controls the error inside the current round. **PIG.EXT** asks a
cross-round question instead: if I stop admitting here and spend one more
forward pass, how much predicted regret on the next sixteen candidates
disappears because they get to see what I just committed? The round closes
when that predicted relief exceeds a budget $$\theta$$. Its two slowest
operating points additionally refuse tokens below a confidence floor.

![Cross-round regret measures the value of waiting]({{ '/images/pig/arxiv_pigext_walk.png' | relative_url }})

*On the same canvas and order, PIG.EXT admits each candidate and then prices
the next ones: how much their mismatch risk would drop if it first committed
what it has (II). The round continues while that predicted relief stays
within $$\theta$$ (III): committing "are" changes little, so the walk widens;
committing "blue" would flip a later slot, so waiting is worth more than
$$\theta$$ and the round closes.*

The practical effect is that PIG adapts its width to the decode. Where the
remaining canvas is flat, both rules commit wide; at the hard spots, where one
commitment would reshape the predictions around it, they narrow down.

![PIG adapts its decoding width to the information curvature]({{ '/images/pig/curvature_width_over_decode.png' | relative_url }})

*Decoding width and local curvature over the decode on the four benchmarks.*

## Results

We evaluate LLaDA-8B-Instruct (256-token canvas, Fast-dLLM's few-shot setup)
and Nemotron-Labs-Diffusion-3B-Instruct (2048-token canvas, 0-shot) on GSM8K,
MATH-500, MBPP and HumanEval, with greedy commits throughout, and report
accuracy against tokens per forward pass (TPF). The baseline is
confidence-threshold decoding over the same eligible positions: Fast-dLLM for
LLaDA, the native confidence decoder for Nemotron.

![Accuracy versus tokens per forward pass on both models]({{ '/images/pig/main_fixedblock.png' | relative_url }})

*PIG.INT (red) and PIG.EXT (blue) against the confidence baseline (grey) on
LLaDA (top) and Nemotron (bottom), fixed blocks. Wilson 95% intervals.*

- **At matched accuracy, PIG.INT reaches up to 1.7× the TPF of Fast-dLLM on
  LLaDA and 1.4× of Nemotron's native decoder.**
- **At the baseline's fastest settings, PIG keeps up to 28 more accuracy
  points.** Confidence thresholds collapse when pushed; a regret budget
  degrades gracefully.
- **It transfers.** The adapter and the tables are fitted once per backbone on
  a small mixture (OpenCoder, GSM8K-train, MATH-train, MBPP-train). Reused
  unchanged on MMLU, a benchmark they never saw, PIG still extends the
  frontier. Against dependency-guided decoders on LLaDA it reaches up to 4.4×
  the TPF.

The paper reduces each comparison to one number in two ways. Table 1 asks
how much faster PIG runs than each baseline at the baseline's best accuracy,
including the dependency-guided decoders DAPD, EB-Sampler and KLASS. Table 2
asks what accuracy each decoder delivers at a fixed throughput.

<div class="table-scroll">
<table class="paper-table">
<caption><strong>Table 1 of the paper: matched-accuracy TPF increase.</strong> Each cell is PIG's TPF over that baseline's best operating point, with PIG's accuracy minus the baseline's in parentheses, at the measured cell of ours that delivers it. Fast-dLLM and both DAPD variants are our own runs; EB-Sampler and KLASS are the cells published by the DAPD authors. LLaDA-8B-Instruct, 64-token fixed blocks.</caption>
<thead>
<tr><th rowspan="2">benchmark</th><th colspan="3" class="grp">our protocol (paired)</th><th colspan="2" class="grp">their protocol</th></tr>
<tr><th>Fast-dLLM</th><th>DAPD-Direct</th><th>DAPD-Staged</th><th>EB-Sampler</th><th>KLASS</th></tr>
</thead>
<tbody>
<tr class="lane"><td colspan="6">PIG.INT: relative TPF increase (accuracy difference in parentheses)</td></tr>
<tr><td>GSM8K</td><td>1.15&times; <span class="d0">(+0.0)</span></td><td>1.77&times; <span class="dp">(+2.0)</span></td><td>2.62&times; <span class="dp">(+0.6)</span></td><td>2.51&times; <span class="dp">(+0.5)</span></td><td>2.71&times; <span class="dp">(+1.0)</span></td></tr>
<tr><td>MATH-500</td><td>1.13&times; <span class="dn">(&minus;0.2)</span></td><td>1.65&times; <span class="dp">(+1.0)</span></td><td>1.82&times; <span class="dp">(+0.6)</span></td><td>3.38&times; <span class="dp">(+1.2)</span></td><td>4.38&times; <span class="dp">(+1.4)</span></td></tr>
<tr><td>MBPP</td><td>1.66&times; <span class="d0">(+0.0)</span></td><td>0.83&times; <span class="dp">(+1.0)</span></td><td>1.85&times; <span class="dp">(+0.4)</span></td><td>2.60&times; <span class="dp">(+3.0)</span></td><td>2.05&times; <span class="dp">(+4.0)</span></td></tr>
<tr><td>HumanEval</td><td>1.29&times; <span class="dn">(&minus;1.8)</span></td><td>1.46&times; <span class="dp">(+2.4)</span></td><td>4.28&times; <span class="dp">(+3.1)</span></td><td>2.88&times; <span class="d0">(+0.0)</span></td><td>3.90&times; <span class="dn">(&minus;0.6)</span></td></tr>
<tr class="lane"><td colspan="6">PIG.EXT: relative TPF increase (accuracy difference in parentheses)</td></tr>
<tr><td>GSM8K</td><td>0.85&times; <span class="dn">(&minus;0.6)</span></td><td>1.04&times; <span class="dp">(+4.4)</span></td><td>1.91&times; <span class="dp">(+1.4)</span></td><td>1.47&times; <span class="dp">(+2.9)</span></td><td>1.59&times; <span class="dp">(+3.4)</span></td></tr>
<tr><td>MATH-500</td><td>0.87&times; <span class="dp">(+0.6)</span></td><td>1.00&times; <span class="dp">(+2.4)</span></td><td>1.57&times; <span class="dp">(+0.8)</span></td><td>2.84&times; <span class="dp">(+2.0)</span></td><td>2.97&times; <span class="dp">(+3.8)</span></td></tr>
<tr><td>MBPP</td><td>1.50&times; <span class="dn">(&minus;0.4)</span></td><td>0.90&times; <span class="dp">(+0.2)</span></td><td>1.73&times; <span class="dp">(+0.8)</span></td><td>2.40&times; <span class="dp">(+1.0)</span></td><td>0.98&times; <span class="dp">(+12.8)</span></td></tr>
<tr><td>HumanEval</td><td>1.46&times; <span class="dn">(&minus;0.6)</span></td><td>1.10&times; <span class="dp">(+6.7)</span></td><td>4.06&times; <span class="dp">(+0.6)</span></td><td>3.25&times; <span class="dp">(+1.2)</span></td><td>4.40&times; <span class="dp">(+0.6)</span></td></tr>
</tbody>
</table>
</div>

<div class="table-scroll">
<table class="paper-table">
<caption><strong>Table 2 of the paper: accuracy at matched TPF.</strong> Accuracies with the gain over Fast-dLLM in parentheses. Every lane is read at exactly the target throughput, linearly between the two measured cells that bracket it. A lane whose sweep ends before the target is not extrapolated: its last measured cell is given instead, as accuracy@TPF. All rows at 64-token fixed blocks.</caption>
<thead>
<tr><th>benchmark</th><th>Fast-dLLM</th><th>PIG.INT</th><th>PIG.EXT</th><th>Fast-dLLM</th><th>PIG.INT</th><th>PIG.EXT</th></tr>
</thead>
<tbody>
<tr class="lane"><td>LLaDA-8B-Instruct</td><td colspan="3" class="grp">interpolated accuracy @ TPF = 10</td><td colspan="3" class="grp">interpolated accuracy @ TPF = 15</td></tr>
<tr><td>GSM8K</td><td>51.4</td><td>70.5 <span class="dp">(+19.1)</span></td><td>56.0 <span class="dp">(+4.6)</span></td><td>25.6</td><td>34.0 <span class="dp">(+8.5)</span></td><td>39.7 <span class="dp">(+14.2)</span></td></tr>
<tr><td>MATH-500</td><td>14.1</td><td>25.5 <span class="dp">(+11.4)</span></td><td>20.0 <span class="dp">(+5.9)</span></td><td>9.8 <span class="at">@12.2</span></td><td>15.4 <span class="at">@13.4</span> <span class="d0">(&ndash;)</span></td><td>13.7 <span class="d0">(&ndash;)</span></td></tr>
<tr><td>MBPP</td><td>21.0</td><td>29.3 <span class="dp">(+8.3)</span></td><td>22.4 <span class="dp">(+1.4)</span></td><td>7.8</td><td>13.2 <span class="at">@15.0</span> <span class="dp">(+5.3)</span></td><td>17.5 <span class="dp">(+9.7)</span></td></tr>
<tr><td>HumanEval</td><td>30.4</td><td>34.3 <span class="dp">(+3.8)</span></td><td>28.7 <span class="dn">(&minus;1.7)</span></td><td>9.2</td><td>20.2 <span class="dp">(+11.0)</span></td><td>19.7 <span class="dp">(+10.5)</span></td></tr>
<tr class="lane"><td>Nemotron-3B-Instruct</td><td colspan="3" class="grp">interpolated accuracy @ TPF = 3</td><td colspan="3" class="grp">interpolated accuracy @ TPF = 5</td></tr>
<tr><td>GSM8K</td><td>86.5</td><td>83.7 <span class="dn">(&minus;2.8)</span></td><td>83.7 <span class="dn">(&minus;2.8)</span></td><td>76.1</td><td>76.7 <span class="dp">(+0.5)</span></td><td>75.9 <span class="dn">(&minus;0.3)</span></td></tr>
<tr><td>MBPP</td><td>38.3</td><td>43.7 <span class="dp">(+5.5)</span></td><td>47.2 <span class="dp">(+8.9)</span></td><td>9.1</td><td>18.8 <span class="at">@4.9</span> <span class="dp">(+9.5)</span></td><td>36.9 <span class="dp">(+27.8)</span></td></tr>
<tr><td>HumanEval</td><td>49.8</td><td>55.8 <span class="dp">(+6.0)</span></td><td>57.0 <span class="dp">(+7.2)</span></td><td>18.8</td><td>22.4 <span class="dp">(+3.6)</span></td><td>44.5 <span class="at">@4.9</span> <span class="dp">(+25.0)</span></td></tr>
</tbody>
</table>
</div>

The adapter is also calibrated, not merely correlated: the predicted mismatch
rate tracks the realised one along the decoder's own trajectories.

![Calibration of the predicted mismatch rate]({{ '/images/pig/phat_calibration.png' | relative_url }})

*Predicted versus realised mismatch on PIG.INT's own trajectories (LLaDA).*

## Try it

Each Hugging Face repository is the whole model: backbone, adapter and tables.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("UT-IFML/LLaDA-8B-Instruct-PIG", trust_remote_code=True)
out = model.generate("Write a Python function that returns the n-th Fibonacci number.",
                     mode="pig.int", phi=0.1)
print(out.text, out.tpf)
```

`mode="pig.ext", theta=0.5` switches rules, `mode="confidence", nu=0.9` gives
the baseline, and `geometry`/`block` set which positions are candidates. The
[repository](https://github.com/Quentin-Fruytier/PIG-DLM-release) has the
evaluation harness that produced every number above.

## What it does not do (yet)

The adapter is trained per backbone: for a new model, harvesting the exact
finite differences took 26 single-H100 GPU hours and fitting the 2M-parameter
adapter about an hour. Commits are greedy; sampling is untouched. The
estimator adds about 13 ms per round in our decoder (measured on a shared
H100, so an upper bound), which is small next to a forward pass of an 8B model
and not nothing for a 3B one. And the padding after the end of a short answer still costs passes
on a fixed canvas, as the strip under the first animation shows, for PIG and
the baseline alike.

## Citation

```bibtex
@article{fruytier2026pig,
  title   = {PIG: Parallel Inconsistency Guidance for Accelerating Diffusion Language Models},
  author  = {Fruytier, Quentin and Xue, Anton and Shakkottai, Sanjay and Sanghavi, Sujay},
  journal = {arXiv preprint arXiv:2610.17375},
  year    = {2026}
}
```
