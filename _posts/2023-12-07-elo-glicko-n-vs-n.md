---
title: "A case against Elo and Glicko for n-vs-n games"
date: 2023-12-07
summary: "A course project: why Elo, Glicko, and Glicko-2 degrade sharply as team size grows in n-vs-n esports — a simulation study, and the case for performance-aware ratings."
math: true
---

*This is a write-up of an earlier course project. The full PDF is linked at the
bottom.*

Rating systems like **Elo** and **Glicko** were designed for **1-vs-1** games —
chess, most famously. But the games with the largest player bases today are
**team-based**: 5-vs-5 titles like Counter-Strike 2, Dota 2, League of Legends,
and Valorant, which mostly run tailored variants of Elo/Glicko under the hood.
This project asks a simple question: **how well do those 1-vs-1 rating systems
actually hold up as the team size $$n$$ grows?** The answer, empirically, is:
*not well at all.*

## The setup

Each player $$i$$ has a latent skill $$\theta_i$$ drawn from a population
distribution (Glicko assumes $$\theta \sim \mathcal{N}(1500, 350)$$), and plays
each game with a noisy performance $$X_i \sim \mathcal{N}(\theta_i, \sigma_i)$$.
For a match of team $$\{1,\dots,n\}$$ vs $$\{n{+}1,\dots,2n\}$$, the standard
extension treats a **team as the sum of its parts**, giving a logistic win
probability

$$P(\text{team wins}) = \frac{1}{1 + \exp\!\big(\tfrac{1}{173.29}(\sum_{j}x_{\text{opp}} - \sum_{i}x_{\text{team}})\big)}.$$

The three algorithms build on each other: **Elo** applies a fixed-$$K$$ update
after each game; **Glicko** adds a *rating deviation* $$RD$$ (a confidence
interval on the rating, so uncertain ratings move faster); **Glicko-2** adds a
*volatility* term to track players who are actively improving. Crucially, all
three see only the **game outcome** — win or loss — not who carried the team.

## The experiment

Simulate a pool of ~20,000 players sampled from $$\mathcal{N}(1500,350)$$, all
initialized at rating 1500. Each round, matchmaking pairs the best against the
best until everyone has a game; outcomes are sampled from the logistic model
above; ratings update via each algorithm. Track the **average percentage rank
error** — how far each player's estimated rank sits from their true rank — as a
function of games played, for different team sizes $$n$$.

## The finding

Accuracy **drops immediately and sharply as $$n$$ grows.** After 1,000 games,
the 1-vs-1 rank error is far below the 5-vs-5 error — and most players play
*fewer* than 1,000 games in a year. Worse, the number of games needed to reach a
fixed error target scales roughly as

$$\text{games to reach } \varepsilon \;\approx\; 2^{\,4n/3},$$

i.e. $$\log_2(\text{games})$$ grows **linearly** in $$n$$. Concretely, a 5-vs-5
system needs on the order of **100× as many games** as 1-vs-1 to reach the same
accuracy. Glicko and Glicko-2 are expected to fail the same way: their
refinements (rating deviation, volatility) don't recover the **individual
performance information** that team-only outcomes destroy.

Two secondary observations: the assumed **Normal** skill distribution is
empirically wrong — real player-rank distributions across popular titles are
clearly **right-skewed** — and the systems have no notion of **game closeness**
(a nail-biter and a stomp move ratings identically).

## The takeaway

The core flaw isn't tuning — it's **information**. Collapsing a team to a single
win/loss throws away exactly the signal you need to rate individuals, and no
amount of Elo/Glicko machinery recovers it. The project argues for a rating
system built for $$n > 1$$ from the start: one that **incorporates individual
in-game performance metrics** (in the spirit of Counter-Strike's HLTV 2.0 or
Leetify ratings) and **game closeness**, rather than bolting team play onto a
1-vs-1 model.

*Full write-up (algorithms, derivations, and figures): [A Case Against Using Elo
or Glicko Algorithms for Rating Players in n-vs-n Games (PDF)]({{ '/data/Online_Learning_Project__Rating_Problem.pdf' | relative_url }}).*
