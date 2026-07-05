---
title: "Ranking Algorithms: A case against Elo and Glicko for n-vs-n games"
date: 2023-12-07
summary: "A course project: why Elo, Glicko, and Glicko-2 degrade sharply as team size grows in n-vs-n esports — a simulation study, and the case for performance-aware ratings."
math: true
---

*This is a write-up of an earlier course project. The full PDF is linked at the
bottom.*

**Elo** and **Glicko** were built for **1-vs-1** games — chess, most famously.
But the games with the largest player bases today are **team-based**: 5-vs-5
titles like Counter-Strike 2, Dota 2, League of Legends, and Valorant, most of
which run tailored variants of Elo/Glicko under the hood. Those ratings aren't
cosmetic — they drive **matchmaking**, which decides whether your games are
balanced or blowouts. So the question this project asks is a practical one:
**how well do 1-vs-1 rating systems actually hold up as the team size $$n$$
grows?** The short answer, from simulation, is *not well at all — and in a way
that gets exponentially worse.*

## The problem: inferring N skills from one bit of feedback

Every player $$i$$ has a latent skill $$\theta_i$$ drawn from a population
distribution (Glicko assumes $$\theta \sim \mathcal{N}(1500, 350)$$). In any
given match, they don't play *at* their skill — they play with a noisy
performance $$X_i \sim \mathcal{N}(\theta_i, \sigma_i)$$, where $$\sigma_i$$ is an
"inconsistency" parameter. The standard way to extend a 1-vs-1 system to teams is
to **treat a team as the sum of its parts**, which gives a logistic probability
that team $$\{1,\dots,n\}$$ beats team $$\{n{+}1,\dots,2n\}$$:

$$P(\text{team wins}) = \frac{1}{1 + \exp\!\big(\tfrac{1}{173.29}(\textstyle\sum_{j}x_{\text{opp}} - \sum_{i}x_{\text{team}})\big)}.$$

(The $$173.29 = 400/\ln 10$$ scale is the usual Elo logistic constant.) The hard
part is now visible: the system observes **one bit** per match — did this team of
$$n$$ players win? — and from that stream of bits it has to recover **$$n$$
separate skill numbers**. As $$n$$ grows, each individual's contribution to the
outcome is increasingly drowned out by their teammates', so each game carries
less and less information about any one player.

## Three algorithms, same blind spot

The project walks through the three canonical systems, each a refinement of the
last:

- **Elo (1978).** After each game, nudge the rating by a fixed step:
  $$r_k \mathrel{+}= K\sum_i\big(s^i_k - \hat P(\text{win})\big)$$. Simple and
  robust, but it *assumes opponents' current ratings equal their true skill*, and
  has **no way to scale the update by how close the game was** — a nail-biter and
  a stomp move your rating identically.
- **Glicko (1995).** Adds a **rating deviation** $$RD$$ — a confidence interval on
  the rating. Uncertain ratings (new or inactive players) move fast; well-
  established ones move slowly. $$RD$$ shrinks with every game played and grows
  during inactivity. This fixes *how confident* we are, but not *what
  information* a game provides.
- **Glicko-2 (2012).** Adds a **volatility** term to track players who are
  actively improving or declining, awarding them more movement. Again: a better
  model of *change*, not of *team attribution*.

Critically, **all three see only the win/loss** — never who carried the team.
The "team = sum of parts" trick keeps the math tractable but throws away exactly
the per-player signal the rating is supposed to estimate.

## The simulation

To measure the damage: sample a pool of ~20,000 players from
$$\mathcal{N}(1500,350)$$, initialize everyone at 1500, and play rounds. Each
round, matchmaking pairs the best against the best until everyone has a game;
outcomes are drawn from the logistic model above; ratings update via Elo. Track
the **average percentage rank error** — how far each player's *estimated* rank
sits from their *true* rank —

$$\mathrm{err}_t = \frac{1}{N}\sum_{i=1}^{N}\frac{\lvert \mathrm{rank}^t_i - \mathrm{rank}_i\rvert}{N},$$

as a function of games played, sweeping the team size $$n \in \{1,2,3,5\}$$.

## The result

![Average rank error vs games played, by team size]({{ '/images/elo-glicko/rank-error-vs-games.png' | relative_url }})

Every curve falls as players accumulate games — but they **plateau at
dramatically higher error the larger the team.** After 1,000 games the average
rank error is ≈ **0.043 for 1-vs-1** but ≈ **0.144 for 5-vs-5** — more than
**3× worse**, and 5-vs-5 is the standard for the most-played competitive titles.
For context, most players play *fewer* than 1,000 games in a year, so in a
team game they may **never** reach a stable, accurate rating.

It's worse than a constant penalty. If you ask how many games it takes to reach a
*fixed* error target ($$\varepsilon = 0.17$$), that count grows so that
$$\log_2(\text{games})$$ is **linear in $$n$$** — i.e.

$$\text{games to reach } \varepsilon \;\approx\; 2^{\,4n/3}.$$

Concretely, a 5-vs-5 system needs on the order of **100× as many games** as
1-vs-1 to hit the same accuracy. And Glicko / Glicko-2 are expected to fail the
same way: their refinements (rating deviation, volatility) address *confidence*
and *change*, neither of which recovers the individual-performance information
that team-only outcomes destroy.

## Two deeper cracks

Beyond the scaling, the simulation surfaces two modeling assumptions that don't
hold up:

1. **The Normal prior is wrong.** Glicko assumes skill is
   $$\mathcal{N}(1500, 350^2)$$, but empirical rank distributions in real titles
   (chess.com, League, CS2, Valorant) are clearly **right-skewed** — a long tail
   of high-skill players the Gaussian can't represent. (Elo's own literature even
   suggested a Maxwell–Boltzmann fit.)
2. **Game closeness is ignored.** A 16–14 nail-biter and a 16–0 stomp produce the
   same rating update. The magnitude of a win carries real information about the
   skill gap, and these systems discard it entirely.

## The takeaway

The core failure isn't tuning — it's **information**. Collapsing a team of $$n$$
players to a single win/loss throws away the per-player signal, and no amount of
Elo/Glicko machinery (bigger $$K$$, rating deviation, volatility) recovers it.
The project argues for a rating system designed for $$n > 1$$ from the start: one
that folds in **individual in-game performance metrics** — in the spirit of
Counter-Strike's HLTV 2.0 or Leetify ratings — and **game closeness**, rather
than bolting team play onto a 1-vs-1 model. That, of course, introduces its own
hard problem: *how do you fairly quantify one player's contribution to a team
outcome?* — which is exactly where a follow-up would go.

*Full write-up (all three algorithms in detail, derivations, and figures): [A Case
Against Using Elo or Glicko Algorithms for Rating Players in n-vs-n Games (PDF)]({{ '/data/Online_Learning_Project__Rating_Problem.pdf' | relative_url }}).*
