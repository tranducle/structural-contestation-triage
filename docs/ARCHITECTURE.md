# Artifact Architecture and Formal Specification

## 1. Prompt-Isolated Telemetry Structural Assessor (PITSA)

Represent an alert as $e = (x_s, x_t)$, where $x_s$ contains categorical structural attributes grouped into $K$ pillars ($x_{s1}, \ldots, x_{sK}$) and $x_t$ contains untrusted attacker-reachable text (such as an endpoint annotation or command line).

PITSA computes the additive marginal log-likelihood ratio:
$$\Lambda(e) = \sum_{k=1}^{K} \log \frac{\widehat{P}(x_{sk} \mid y=1)}{\widehat{P}(x_{sk} \mid y=0)}$$
using Laplace-smoothed empirical marginals fitted on the training partition. The binary gate decision is $g(e) = \mathbf{1}[\Lambda(e) \ge \tau]$, where $\tau = 0.72$ (`0.719172`) is selected on the SALAD validation partition.

- **Proposition 1 (Text-channel invariance)**: Because neither $x_t$ nor any text-derived feature enters $\Lambda(e)$ or the fitted tables, $\Lambda(e) = \Lambda(e')$ and $g(e) = g(e')$ for every text-only manipulation $x_t \to x_t'$.

## 2. Gate-Bounded Contestation Policies

Let $a \in \{0, 1\}$ be the language model's opening disposition (`1` = escalate, `0` = dismiss; unparseable responses default to `1` under the fail-safe rule), and let $b \in \{0, 1\}$ be the second-turn disposition issued **only** when $g = 1$ and $a = 0$:

- **`PITSA` (Gate alone)**: $d_{\mathrm{PITSA}} = g$
- **`U` (Unassisted model)**: $d_U = a$
- **`D` (Spotlighting)**: $d_D = a^D$, where untrusted text is wrapped in `<untrusted_annotation>` tags with explicit data-only instructions.
- **`S` (Structural contestation)**: $d_S = a + g(1 - a)b$, where the second turn reports $\Lambda(e)$, $\tau$, and the top two signed pillar contributions.
- **`P` (Placebo second turn)**: $d_P = a + g(1 - a)b^P$, where the second turn requests a quality-check re-review without structural evidence.
- **`M` (Matched placebo second turn, extension)**: $d_M = a + g(1 - a)b^M$, matching the structural cue's length, statement of prompt-isolated check, and review flag while omitting the numerical score, threshold, and pillar contributions.
- **`DS` (Spotlighting + structural contestation)**: $d_{DS} = a^D + g(1 - a^D)b^{DS}$.
- **`G` (Model-or-gate rule)**: $d_G = \max(a, g)$.

- **Proposition 2 (Bounded contestation)**: Pointwise for every alert, $d_U \le d_S \le d_G$, so $R_U \le R_S \le R_G$, $F_U \le F_S \le F_G$, and $F_S - F_U \le \Pr(g = 1, a = 0 \mid y = 0) \le \Pr(g = 1 \mid y = 0)$.

## 3. Expected Review Cost Model

With each escalation costing `1` analyst review, each missed hostile alert costing $k \ge 1$ reviews, and stream hostile prevalence $\pi \in (0, 1)$, expected review cost per alert is:
$$L = \pi(1 - R)k + \pi R + (1 - \pi)F$$
