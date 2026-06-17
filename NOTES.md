# Engineering notes (PTCGABC / cabt)

Living doc of what the actual installed env exposes, paired with the HANDOVER
items that need correcting and the path to a learned policy.

## 2026-06-17 update — Python wrappers exist!

The HANDOVER's `cabt.api` Python wrappers **do exist** — they're shipped
inside the official `sample_submission/cg/` bundle (downloadable via
`kaggle competitions download -c pokemon-tcg-ai-battle`, copied into this
repo's `cg/`). They are NOT in `pip install kaggle-environments` for
historical reasons, but they ARE in the submission runtime as long as we
include `cg/` in our tar.gz.

This unfreezes the PIMC / IS-MCTS path that previous notes marked as
"drop entirely". `cg.api` (see line refs to the in-repo file) exposes:

- Dataclasses: `Observation` (`cg/api.py:439`), `SelectData` (`:399`), `Option` (`:382`),
  `State` (`:367`), `PlayerState` (`:351`), `Pokemon` (`:339`), `Card` (`:333`),
  `SearchState` (`:448`), `Skill` (`:459`), `CardData` (`:464`), `Attack` (`:485`)
- Enums: `OptionType` 0-16 (`cg/api.py:120`), `SelectType` 0-10 (`:55`),
  `SelectContext` (`:68`), `AreaType` 1-12 (`:11`), `EnergyType` 0-11 (`:25`),
  `CardType` 0-6 (`:39`), `SpecialConditionType` 0-4 (`:48`), `LogType` (`:189`)
- `all_card_data() -> list[CardData]` (`cg/api.py:495`)
- `all_attack() -> list[Attack]` (`:502`)
- `to_observation_class(obs_dict) -> Observation` (`:509`)
- `search_begin(agent_observation, your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active, manual_coin=False) -> SearchState` (`:517`)
- `search_step(search_id, select) -> SearchState` (`:597`)
- `search_end()` (`:629`), `search_release(search_id)` (`:633`)

Battle entrypoints in `cg/game.py`: `battle_start(deck0, deck1)` (`:19`),
`battle_select(select_list)` (`:48`), `battle_finish()` (`:43`),
`visualize_data() -> str` (`:69`).

Official API docs: <https://matsuoinstitute.github.io/cabt/>

## VERIFY answers (HANDOVER §5) — corrected

1. **Import path.** Two flavors:
   - **For self-play / training** (this image): `kaggle_environments.envs.cabt.cg.game`
     and `...cg.sim`. ABI here is sufficient for `battle_start/select/finish`.
   - **For submission** (Kaggle runtime): include `cg/` from
     `sample_submission/` in the tar.gz; import as `from cg.api import ...`,
     `from cg.game import battle_start, ...`. The `kaggle_environments`
     package is NOT guaranteed at submission time, but `cg/` is whatever
     we bundle.
2. **Terminal & value.** `obs["current"]["result"]` is `-1` while playing,
   `0` if P0 wins, `1` if P1 wins, `2` for draw. `env.steps[-1][i].reward`
   exposes `+1 / 0 / -1` after termination.
3. **Search rewind.** Now testable via `cg.api.search_begin`. Hidden info
   (opponent hand/deck/prize/active) is provided as `list[int]` (card IDs)
   to the call; engine validates lengths against the visible
   `deckCount`/`handCount`/`prize`. Use `search_release(id)` to free state.
4. **`search_begin_input`.** Plain ASCII ~80 chars, produced by
   `GetBattleData` and passed back into `SearchBegin` via `cg.api`.
5. **`legal_pool`.** The contest-legal cards are listed in
   `kaggle_data/EN_Card_Data.csv` / `JP_Card_Data.csv` (downloaded with the
   competition data). Each row has Card ID, name, expansion, etc.
6. **`deckCount` vs `len(deck)`.** Unchanged — visible counts match array
   lengths.

## Observation cheat sheet (empirical)

```
obs = {
  "select": None | {
    "type":      int,       # SelectType
    "context":   int,       # SelectContext
    "minCount":  int,
    "maxCount":  int,
    "option":    list[dict] # each option has {"type": OptionType, ...}
    "deck":      list[dict] | None,
    "remainEnergyCost":     int,
    "remainDamageCounter":  int,
    "contextCard":          dict | None,
    "effect":               dict | None,
  },
  "logs":   list[dict],     # increment since last call
  "current": None | {
    "turn": int, "turnActionCount": int,
    "yourIndex": 0|1, "firstPlayer": -1|0|1,
    "supporterPlayed": bool, "stadiumPlayed": bool,
    "energyAttached": int, "retreated": bool,
    "result": -1|0|1|2,
    "stadium": list, "looking": list,
    "players": [P0, P1]      # see below
  },
  "search_begin_input": str  # opaque, see VERIFY #4
}
```

PlayerState keys: `active, bench, benchMax, deckCount, discard, prize,
handCount, hand, poisoned, burned, asleep, paralyzed, confused`. `hand` is
`None` for the opponent.

### Decision distribution in self-play (mirror sample deck)

```
(select.type, select.context)  count
(0, 0)   MAIN/MAIN               73
(1, 3)   CARD/DISCARD            12
(1, 7)   CARD/TO_BENCH(?)        11
(1, 22)  CARD/...                 4
(1, 4)   CARD/SWITCH              3
(1, 8)   CARD/...                 3
(1, 1)   CARD/SETUP_ACTIVE        2
(1, 2)   CARD/SETUP_BENCH         2
(9, 41)  YES_NO/IS_FIRST          1
(8, 38)  NUMBER/...               1
```

### OptionType full enumeration (cg.api.OptionType, official)

```
 0  NUMBER            {"number": int}
 1  YES               {}
 2  NO                {}
 3  CARD              {"area", "index", "playerIndex"}
 4  TOOL_CARD         {"area", "index", "playerIndex", "toolIndex"}
 5  ENERGY_CARD       {"area", "index", "playerIndex", "energyIndex"}
 6  ENERGY            {"area", "index", "playerIndex", "energyIndex", "count"}
 7  PLAY              {"index"}
 8  ATTACH            {"area", "index", "inPlayArea", "inPlayIndex"}
 9  EVOLVE            {"area", "index", "inPlayArea", "inPlayIndex"}
10  ABILITY           {"area", "index"}
11  DISCARD           {"area", "index"}
12  RETREAT           {}
13  ATTACK            {"attackId"}
14  END               {}
15  SKILL             {"cardId", "serial"}
16  SPECIAL_CONDITION {"specialConditionType"}
```

Mirror baseline self-play hit 7/8/9/13/14 frequently. 10-12, 15-16 are
emitted in non-mirror match-ups (abilities, retreats, layered effect ordering).

## Files

```
main.py               # Kaggle submission entry point (renamed from agent.py).
                      # Reads deck.csv at startup, loads train/policy.npz if
                      # present, otherwise falls back to engine-order prior.
agent.py              # Local-dev legacy entry. Kept because selfplay_test.py
                      # and train/reinforce.py still import from it.
deck.csv              # 60 card IDs, one per line. Submitted as-is.
cg/                   # Official Python wrappers + libcg.so + cg.dll. Bundled
                      # at the root of submission.tar.gz.
make_submission.sh    # ./make_submission.sh -> submission.tar.gz
selfplay_test.py      # python3 selfplay_test.py [N]  → benchmark vs random
train/
  features.py         # obs -> state/option feature vectors (36-d each)
  policy.py           # Linear policy (numpy), save/load .npz
  reinforce.py        # python3 -m train.reinforce --episodes N
  policy.npz          # trained weights (created by training)
kaggle_data/          # Extracted contents of pokemon-tcg-ai-battle.zip
  EN_Card_Data.csv    # Card master (English)
  JP_Card_Data.csv    # Card master (Japanese)
  sample_submission/  # Reference impl (use cg/ from here)
```

## Training quick start

`scripts/run.sh` wraps the python invocation with the `LD_LIBRARY_PATH`
adjustments numpy / kaggle_environments / torch need on this nix image.

```bash
# Warm-started 2000ep training, ~6 min on CPU.
scripts/run.sh python3 -m train.reinforce \
    --episodes 2000 --lr 0.05 \
    --warm-start train/policy.npz \
    --out train/policy.npz \
    --metrics-out train/metrics_2000ep.json

# Benchmark vs random.
scripts/run.sh python3 selfplay_test.py 20
```

Current setup: numpy-only linear policy on 40-d state + 40-d option features
(`train/features.py`), REINFORCE with terminal reward (`train/reinforce.py`).
State features include super-effective matchup flags and retreat costs
sourced from `cg.api.all_card_data()`. Cumulative 5000ep training (2000ep
from scratch + 3000ep warm-start at lr 0.03) beats `random_agent` 72-8
(90%) over 80 games. Win rate is saturating against random_agent at this
feature complexity; further structural gains likely need MLP or PIMC.

### Upgrade path

1. **Search (UNFROZEN 2026-06-17; framework lands but doesn't move the needle yet).**
   `cg.api.search_begin/step/release` is wired through `train/pimc.py` and
   gated by `POKEAI_PIMC=1` in `main.py`. After three backend iterations
   (1-ply value head → single-world rollout → multi-world IS-MCTS), the
   definitive 100-game numbers came out as a tie:

     PIMC-ON vs PIMC-OFF mirror:  51-49 (51%)  Wilson [41%, 61%]
     PIMC-ON vs random:           92-8  (92%)  Wilson [85%, 96%]
     PIMC-OFF vs random:          91-9  (91%)  Wilson [84%, 95%]

   So PIMC-ON gives statistically equivalent strength to PIMC-OFF at
   24x the per-move latency. The earlier "PIMC-ON wins" / "PIMC-ON
   loses" results at 20-40 games were all within sample variance. The
   framework remains as scaffolding — search_begin / search_step
   semantics, multi-world Q averaging, engine-prior rollout option —
   but a real improvement needs a stronger rollout policy or a
   non-linear (MLP/PyTorch) value head before turning PIMC on by default.
2. **Better features.** Add card-id embeddings (look up `Pokemon.id`,
   energy types, attack costs) and tile-encoded counts (active/bench HP
   per slot). `all_card_data()` is the canonical source.
3. **Bigger model.** Drop in a small MLP (PyTorch) once the feature dim
   stops being the bottleneck. `torch==2.11.0+cu128` is already in `.venv`
   and `scripts/env.sh` resolves `libcuda.so.1` via `/usr/lib/wsl/lib`,
   so `torch.cuda.is_available()` returns True on the RTX 3070 Ti.
4. **PPO + value baseline.** Replace `reinforce_update` with a clipped
   PPO objective and a value head; that removes the high-variance terminal
   reward signal that REINFORCE suffers from.
5. **Self-play league.** Keep a rotating snapshot of past policies and
   train against the league instead of only the current policy — prevents
   cycle-collapse where the agent over-fits to its own quirks.

## Seed-selection log for ensemble members (2026-06-17)

To avoid the seed=1024 regression (a sub-50% solo member that dragged
the ensemble down), every new candidate gets a 40-game solo bench
vs the linear policy before being added to the ensemble. Above 50%:
include. Below 50%: discard.

Running tally:

  seed=20260628 (first MLP):     ~57.5% vs linear  ← included
  seed=42 (second MLP):          (in 2-MLP since first day, never solo-checked
                                  but the 2-MLP ensemble strictly beat single)
  seed=1024:                     45%   vs linear   ← rejected
  seed=100:                      52.5% vs linear   ← included (3-MLP)
  seed=200:                      47.5% vs linear   ← rejected

So 2 of 4 new candidates passed. The bench is a noisy 40-game proxy
but it's cheap to run and matches our "submission-budget" risk
appetite — we'd rather hold a verified-strong ensemble than gamble
slots on an untested candidate.

Discarded seeds are stored in /tmp/ (not in train/) so the ensemble
glob loader doesn't pick them up. The metrics JSON is committed so
the failure record is preserved.

## 3-MLP ensemble adoption (2026-06-17) — current submission default

Added a third MLP member to the ensemble. Per the lesson from the
earlier failed seed=1024 attempt (which had solo mirror winrate
9-11 = 45% vs linear and dragged the ensemble down), this time we
benched the new seed alone first.

Trained 2000ep self-play at --lr 1e-3, seed=100, ~9 min on RTX 3070 Ti.

Solo bench (40 games):
  seed=100 vs linear:        21-19 (52.5%)  — passes the >50% threshold
  seed=100 vs random:        37-3  (92.5%)
  seed=100 vs rule_based:    7-33  (17.5%)  — single MLPs are bad here

Marginal but above-average; kept the seed in. 3-MLP ensemble bench
across all 4 Kiyota meta agents (30 games each, larger sample than
the 2-MLP 20-game baseline):

                       2-MLP (20 games)    3-MLP (30 games)    Δ
  vs Mega Lucario:     35.0% (7-13)        20.0% (6-24)        -15pp
  vs Dragapult:        15.0% (3-17)        33.3% (10-20)       +18.3pp
  vs Iono's:           10.0% (2-18)        13.3% (4-26)        +3.3pp
  vs Mega Abomasnow:   30.0% (6-14)        40.0% (12-18)       +10pp
  overall:             22.5% (18-62)       26.7% (32-88)       +4.2pp

Wilson 95% CIs still overlap (22.5% [15-33] vs 26.7% [19-35]) so this
isn't statistically conclusive, but the point estimate trends up
across 3 of 4 match-ups. The Mega Lucario regression makes sense —
seed=100 doesn't share whatever quirk made the 2-MLP ensemble peak at
35% there, so its inclusion pulls the average down on that one
match-up while pulling the other three up.

main.py picks up all mlp_policy*.pt files via glob, so the addition
needs no code change. Submission tar.gz adds 39 KB; verification by
the new check_main_exec hook passes.

Not yet submitted — saving the remaining Kaggle daily slots for a
clearer signal. Current LB submission (53776818) is the 2-MLP and
sits at score ~633 after 4 PUBLIC episodes (2W/2L).

## MLP ensemble (2026-06-17) — earlier 2-MLP iteration

Single 2000ep MLP beat linear (57.5% mirror) but regressed vs rule_based
(22.5% vs linear's 30%). The diagnosis was overfitting to the self-play
distribution — same pattern as the earlier vs-opp finetune attempts.
Standard cheap fix: train independently-seeded MLPs and average their
logits.

train/ensemble_policy.py: EnsemblePolicy with the same .logits() /
.value() / .probs() API as MlpPolicy. Averages logits across N members
(np.mean over stacked logits). Auto-loaded by main.py when 2+
mlp_policy*.pt files exist in train/.

Trained a second 2000ep MLP at seed=42 (~9 min, indistinguishable
mirror-self-play dynamics from seed=20260628).

A/B (40 games):
                          Single MLP    Ensemble (2 MLPs)    Δ
  vs random:              38-2 (95.0%)  37-3 (92.5%)        -2.5pp (noise)
  vs rule_based(Lucario):  9-31 (22.5%) 13-27 (32.5%)       +10pp
  ensemble vs single MLP:   —           23-17 (57.5%)       (sanity check)

The +10pp vs rule_based is the headline: averaging across seeds
cancels each MLP's per-seed overfit quirks. Vs random the small drop
is within the 40-game CI. Mirror match against the single MLP shows
the ensemble genuinely picks different moves (it's not just one
member overpowering).

main.py decision stack updated to try EnsemblePolicy first (when 2+
mlp_policy*.pt files exist), then single MLP, then LinearPolicy.
make_submission.sh / check_bundle.py bundle both .pt files; tar.gz
stays at ~1.1 MB.

The next step in this direction would be a third seed, then a fourth.
Each additional member should cost less than the first (diminishing
returns) but the per-move latency stays linear in N — at N=2 we add
maybe 30ms per move which is well within the budget.

## Experiment: 3-MLP ensemble (2026-06-17) — seed 3 dragged the ensemble down

Hypothesis: 2-MLP ensemble already gave +10pp vs rule_based; a third
seed should add a touch more variance reduction.

Trained a third 2000ep MLP at seed=1024 (9 min on RTX 3070 Ti). The
training metrics looked identical to the other two (mirror winrate
0.55-0.62, no apparent overfitting).

3-MLP ensemble bench (40 games each):
                           2-MLP         3-MLP        Δ
  vs random:               92.5%         90.0%       -2.5pp
  vs rule_based(Lucario):  32.5%         17.5%       -15.0pp  ← worse than single MLP
  ensemble vs main.agent:    —           50.0%       (mirror, expected)

The 3-member ensemble actually fared WORSE vs rule_based than the
2-member ensemble. To diagnose, ran the seed-3 MLP alone on 20 games:
  seed-3 alone vs linear:        45%   (seed-1 alone was 57.5%)
  seed-3 alone vs rule_based:    20%
  seed-3 alone vs random:        95%

Seed 3 is a genuinely weaker model — its mirror winrate is below
50% against the linear policy. Adding a weak model to the average
pulled the ensemble's choices toward its biases.

Diminishing returns turned out to be a misdiagnosis; the issue is
selection bias. The 2-MLP ensemble worked because BOTH members were
above-average. Random seed 1024 happened to give a below-average
member, and naive averaging let it vote.

Moved train/mlp_policy_seed3.pt out of train/ so main.py keeps the
2-MLP setup. The metrics file stays in tree as the experiment record.

Lessons:
  - When training ensemble members, validate each member's individual
    strength before including it.
  - Future ensemble work should consider weighted averaging (members
    weighted by their solo win rate) or member dropout (remove the
    weakest at eval time).
  - Or the simpler version: train N seeds, keep the top K by solo
    mirror winrate against the linear baseline.

## Experiment: extend MLP self-play to 5000ep (2026-06-17)

Hypothesis: 2000ep was enough to beat the linear policy (57.5% mirror)
but maybe more episodes would consolidate the win and push vs-random
higher. Warm-started from the 2000ep MLP, +3000ep self-play, --lr 5e-4
(half of the original to limit drift), seed 20260630, 13.5 min on RTX
3070 Ti. Self-play winrate stayed in the 0.55-0.65 band (healthy mirror
dynamics, no extreme P0 takeover).

A/B (40 games each):
                                  2000ep MLP    5000ep MLP    Δ
  vs linear(ours):                23-17 (57.5%) 16-24 (40.0%) -17.5pp
  vs rule_based(Lucario):          9-31 (22.5%)  6-34 (15.0%) -7.5pp
  vs random:                      38-2  (95.0%) 39-1  (97.5%) +2.5pp

Wilson 95% CIs do overlap pairwise (small sample), but the point
estimates move uniformly in the wrong direction for the two stronger
opponents while gaining tiny ground vs random. That's the classic
overfit signature: more episodes squeezed marginal vs-random
performance at the cost of the generalization that beat linear.

Rolled back to 2000ep MLP. metrics_mlp_5000ep.json stays as the
experiment record so future iterations don't re-run the same loop.

Implication for the linear-style "more training is always better"
mindset: with an MLP, more episodes are not free — the policy
distribution narrows toward whatever pattern wins self-play, and
that pattern stops being robust to opponents off the self-play
distribution. The fix is exposure to different opponents during
training (something rule-based finetune tries but its V-head OOD
problem ate the gains), or KL regularization back to an earlier
checkpoint.

## Experiment: MLP vs-rule-based fine-tune with advantage baseline (2026-06-17)

Hypothesis: the MLP regressed against rule_based (22.5% vs linear's 30%)
because it overfits to the self-play state distribution. Fine-tuning
against rule_based with the advantage baseline (which we proved fixes
the linear regression) should close the gap.

Setup: scripts/train_mlp_vs_opponent.py wraps the MLP REINFORCE update
with a fixed opponent and the advantage baseline. Warm-started from
the 2000ep self-play MLP, 1500ep vs rule_based, --lr 5e-4 (lower than
self-play's 1e-3 to slow drift), seed 20260629, 7 min wall-clock.
Cumulative training winrate: 331/1500 = 22%.

A/B (40 games):
                              pre-finetune    post-finetune
  mlp vs linear:              23-17 (57.5%)  21-19 (52.5%)  -5pp
  mlp vs rule_based(Lucario): 9-31  (22.5%)  8-32  (20.0%)  -2.5pp
  mlp vs random:              38-2  (95.0%)  36-4  (90.0%)  -5pp

All three benches regressed. The MLP got slightly worse vs rule_based
(the target), and noticeably worse vs random and vs the linear policy.

Interpretation: fine-tuning against the strong opponent moved the
policy AWAY from the self-play equilibrium it was strong at, without
finding a robust strategy against rule_based. The MLP's value head was
trained on self-play states, so V(state) during the rule_based games
is unreliable — the advantage subtraction is mostly working with the
raw -1 signal, the same failure mode the linear policy showed.

Rolled back to the pre-finetune MLP. scripts/train_mlp_vs_opponent.py
and the metrics file stay in tree. The next attempt to close the
rule_based gap probably needs a two-stage fine-tune: (a) freeze the
policy head and train the value head on vs-rule-based rollouts until
V(state) is calibrated, then (b) unfreeze and apply policy gradient.
Or simpler: just accept that vs strong opponents we'll lose 70-80%
and focus on the rating math (more wins vs weak opponents).

## PyTorch MLP policy (2026-06-17) — adopted as submission default

Built `train/mlp_policy.MlpPolicy`, a small Adam-trained MLP with the
same .logits() / .value() / .probs() API as LinearPolicy:

  policy head: concat(state, option) -> Linear(80, 64) -> ReLU
                                     -> Linear(64, 32) -> ReLU -> Linear(32, 1)
  value head:  state -> Linear(40, 32) -> ReLU -> Linear(32, 1) -> tanh
  param count: 8642
  device:      cuda when available, else cpu

Engine-order bias preserved (`b_order * (n-1-i)/(n-1)` per option) so
an untrained MlpPolicy still matches first_agent.

train/mlp_train.py wraps the REINFORCE + value MSE update with PyTorch
autograd. Always uses the advantage baseline. Trained 2000ep self-play
(--lr 1e-3, seed 20260628, 9 min on RTX 3070 Ti).

A/B (40 games each):
  mlp vs linear(ours):        23-17 (57.5%) ← MLP beats linear in mirror
  mlp vs random:               38-2 (95.0%) (linear baseline: 91% over 100 games)
  mlp vs rule_based(Lucario):  9-31 (22.5%) (linear baseline: 30%)

So the MLP is strictly stronger than the linear policy at self-play
mirror, marginally better vs random, and worse vs rule_based. The mirror
result is the most reliable: 57.5% on 40 games has Wilson 95% CI
[42%, 71%], just above 50% which means the MLP genuinely picks
better moves than the linear policy can express.

The rule_based regression mirrors what we saw with vs-opp training: a
more expressive model overfits to self-play distribution and gets
confidently-wrong on out-of-distribution states. Future work could
warm-start the MLP from vs-rule-based rollouts, or use a deeper value
head to detect uncertain states.

main.py tries `MlpPolicy.try_load()` first, falls back to
`LinearPolicy.try_load()` if torch is unavailable or the .pt file is
missing. The submission tarball now bundles both train/policy.npz and
train/mlp_policy.pt (+38 KB to the 1.06 MB submission).

## Experiment: train our linear policy on Mega Lucario deck (2026-06-17)

Hypothesis: the rule-based + Mega Lucario combo beats us partly because
the deck is just stronger; if we retrain our linear policy on that same
deck we should at least catch up.

Setup: swapped deck.csv to deck_mega_lucario.csv (60 cards, card 6 x13
basic Fighting energy), updated train/reinforce.py to read deck.csv at
import time instead of importing DECK from agent.py, and trained on the
Lucario deck.

### Run 1: 3000ep from scratch (--lr 0.05, --lr-value 0.05, seed 20260624)
17 min wall-clock; |w_opt| 0 -> 1.67.

  new linear(Lucario) vs rule_based(Lucario): 15-25 (37.5%)
  new linear(Lucario) vs random:               30-10 (75.0%)
  old linear(our deck) vs random:              91% (100-game definitive)

### Run 2: +3000ep warm-start (--lr 0.03, --lr-value 0.05, seed 20260625)
17 min wall-clock; |w_opt| 1.67 -> 2.26. Cumulative 6000ep.

  6000ep linear(Lucario) vs rule_based(Lucario): 13-27 (32.5%) (regressed from 37.5%)
  6000ep linear(Lucario) vs random:               25-15 (62.5%) (regressed from 75%)

So MORE training on Mega Lucario actually made things WORSE on the
40-game vs-random bench. Possible reasons:
  - The features we built for our deck (option's attack damage / cost,
    super-effective matchups using card_data) don't capture the
    play-pattern of Mega Lucario (which leans hard on evolution chains
    and tool / supporter combos that our policy can't represent).
  - Mega Lucario games are visibly longer (2x training time) and the
    REINFORCE terminal-reward signal gets noisier per step.
  - lr=0.03 in the warm-start may have been too small to keep moving
    after the initial 3000ep already converged into a local optimum
    that the new feature basis can't escape.

Net: Mega Lucario direction is a dead end for our linear policy at
this feature complexity. Rolled back to old deck + old policy.npz
(95% vs random on 20 games, matching the 91% on the 100-game baseline).

Kept train/metrics_{3000,6000}ep_lucario.json as experiment records.
The reinforce.py refactor to read deck.csv at runtime stays — it's
independently useful.

## Experiment: REINFORCE vs the rule-based agent as opponent (2026-06-17)

Hypothesis: self-play converges to mirror-match equilibrium with no
incentive to learn moves that beat a different opponent. Training
against the strong rule-based agent should force the policy to learn
moves that punish its mistakes.

Setup: scripts/train_vs_opponent.py runs REINFORCE where only our side
is recorded and updated; opponent is a fixed black-box agent (rule_based
or random). Warm-started from the 5000ep policy, 2000ep vs the rule-based
Mega Lucario agent (--lr 0.05 --lr-value 0.05, seed 20260626, 7 min).

Result (40 games):
  warm-start linear vs rule_based:    28-12 lose (was 30%, became 6-34 = 15%)
  vs-rule-trained linear vs rule_based: 6-34 (15%)  ← worse
  vs-rule-trained linear vs random:     35-5 (87.5%) ← slight regression

So training against a strictly stronger opponent made us **worse**, not
better. Cumulative rule_based training showed 380 wins / 1619 losses =
19% — most episodes ended with reward = -1.

Root cause: reinforce_update has no advantage baseline. The gradient is
`reward * (of_picked - expected_of)`. When reward = -1 dominates (as it
does against a strong opp), every update pushes the policy away from the
move it just sampled, toward uniform. The policy regresses toward
maximum-entropy babble, not toward the rare winning moves.

Fix would be to use advantage `(reward - V(state))` instead of raw reward,
which `policy.value(state)` from the value head naturally supplies. Held
off implementing because the experiment also revealed that even at 380
wins out of 2000 episodes the signal-per-state is too sparse for this
linear-feature setup to actually learn a winning strategy against a
hand-coded card-aware agent.

Rolled back the policy. scripts/train_vs_opponent.py and the metrics file
stay in tree as infrastructure for the next attempt (which needs the
advantage baseline before re-running this loop).

### Follow-up: vs rule-based REINFORCE WITH advantage baseline (2026-06-17)

reinforce_update gains a `use_advantage` flag. When True, the policy
gradient is scaled by `reward - V(state)` instead of raw `reward` —
V(state) comes from the existing value head (trained alongside the
policy in self-play). train_vs_opponent.py defaults --use-advantage on.

Same setup as above: warm-start from 5000ep policy, 2000ep vs rule_based,
--lr 0.05 --lr-value 0.05, seed 20260627. 7 min wall-clock. |w_opt|
changed less (2.64 -> 2.80 vs 2.64 -> 3.23 without advantage) — the
baseline correctly moderates the gradient when V(state) already explains
the loss.

A/B (40 games):
  with-advantage vs rule_based:  8-32 (20%)  (was 30% baseline, 15% no-adv)
  with-advantage vs random:      38-2 (95%)  (was 91% baseline, 87.5% no-adv)

The advantage baseline prevented the vs-random regression — that
confirms the diagnosis. But vs rule_based we still drift from 30% to
20% because the value head was trained on self-play states, not on
states reached against rule_based, so V(state) is unreliable in those
games and the advantage is mostly still raw reward.

Trade-off: keeping this policy gives ~+4pp vs random (within noise) at
~-10pp vs rule_based (statistically meaningful). For a competition
matched against players similar to rule_based, the swap is a net loss.
Kept the 5000ep self-play policy as the submission default.

Future fix: pre-train V(state) on vs-rule-based rollouts before
running the policy gradient loop, or use a deeper (MLP) value head.

## Experiment: multi-opponent training across 4 meta decks (2026-06-17)

Hypothesis: single-opp finetune kept overfitting to one opponent's
patterns. Sampling uniformly across the 4 Kiyota meta agents per
episode should give a diverse-enough gradient signal that the policy
can't collapse onto a single counter.

scripts/train_mlp_vs_meta.py rotates Mega Lucario / Dragapult /
Iono / Mega Abomasnow uniformly, always uses advantage baseline,
gradient-clipped at 1.0. Per-opponent winrate logged so we can see
which match-up is dragging the signal.

Warm-started from the 2000ep self-play MLP, 2500ep multi-opp,
--lr 5e-4, seed 20260701, 13 min. End-of-training recent winrates
(last 100 episodes): mega=52, drag=25, iono=7, abom=32, overall ~33%.

A/B with the meta-trained MLP ALONE (not in the ensemble) — 20 games
per opponent:
                       pre (orig 2-MLP)   post (meta-trained alone)
  vs Mega Lucario:     35%               5%       -30pp
  vs Dragapult:        15%               15%       0pp
  vs Iono's:           10%               15%      +5pp
  vs Mega Abomasnow:   30%               30%       0pp
  overall:             22.5%             16.2%    -6pp

Trying it inside a (meta-trained + seed2) ensemble:
                       overall vs meta:  22.5% (no change from original)

So meta finetune actively HURTS the policy on its own and gives zero
ensemble lift either. The signal is too sparse (each opp only seen
~625 times) and the policy can't simultaneously improve across all
four match-ups with the same parameters — what helps Iono hurts Mega
Lucario.

Moved train/mlp_policy_meta.pt out of the train/ directory so the
ensemble auto-loader picks up only the 2 healthy MLPs.
metrics_mlp_vs_meta_2500ep.json stays as the experiment record.

Failure modes encountered so far (all logged here for future
iterations): single-opp finetune (overfit), 5000ep extension
(self-play overfit), MLP vs-rule-based finetune (V-head OOD),
3-MLP ensemble with bad seed (selection bias), and now uniform
multi-opp training (signal too sparse to improve everywhere).

Future direction probably needs either (a) MUCH more training
data per opponent (10k+ episodes per opp), (b) opponent-conditioned
policy (policy takes an opponent-id input so it can specialize per
match-up), or (c) accept the 22.5% floor and ship.

## Meta-deck matchup table (2026-06-17)

Added three more Kiyota meta-deck rule-based agents alongside the
existing Mega Lucario: Dragapult ex, Iono, Mega Abomasnow. Each is the
same shape as scripts/rule_based_agent.py (vendored from a Kaggle
notebook with attribution in the file header, ruff-excluded). Each has
its own deck under deck_<name>.csv and reads it via a per-agent
RULE_DECK_PATH_<NAME> env var (default falls back to the .csv next to
the script's parent directory).

scripts/bench_meta.py runs the current submission against all four
agents. 20-game results:

  main.agent vs Mega Lucario:   7-13 (35.0%)
  main.agent vs Mega Abomasnow: 6-14 (30.0%)
  main.agent vs Dragapult ex:   3-17 (15.0%)
  main.agent vs Iono's:         2-18 (10.0%)
  overall (80 games):           18-62 (22.5%)

The 95% vs random and 57.5% mirror-vs-linear numbers we'd been
tracking were misleading: against the actual meta agents the
2-MLP ensemble loses overall ~77%. Mega Lucario is the matchup we
do best in; Iono's deck (heavy Lightning-energy spam with Wattrel
chains) is the worst.

This is the rating-relevant evaluation table. Future strength
benches should run scripts/bench_meta.py at higher game counts
(40-80) instead of just vs random / vs single rule-based.

## External baselines we benchmarked

`scripts/rule_based_agent.py` and `deck_mega_lucario.csv` are vendored from
the Kaggle reference notebooks (see file header for source URLs). They
were saved to give us a concrete strong-baseline opponent for our own
work. 40-game results on 2026-06-17:

  rule_based(Mega Lucario) vs linear(ours, our deck):  28-12 (70%) ← rule-based wins
  rule_based(Mega Lucario) vs random_agent:             40-0  (100%)
  linear(ours, our deck) vs random_agent:               34-6  (85%)
  linear(ours, Mega Lucario deck) vs random_agent:      29-11 (72.5%)

Two takeaways:
  - Just swapping our deck to Mega Lucario degrades our linear policy
    (72.5% vs 85%) because the policy was trained on a different deck and
    its option-side features don't generalize to the new card pool.
  - Rule-based + Mega Lucario beats our entire pipeline. For a stronger
    submission we'd need either to wrap the rule-based agent (cheap; loses
    our learning infrastructure) or to retrain the linear policy on the
    Mega Lucario deck (medium effort; uncertain gain).

These resources stay in tree as reference / opponent. We aren't shipping
the rule-based code as our submission yet — see commit message for the
decision rationale.

## Open items

- `_try_load_policy()` silently swallows exceptions to keep the Kaggle
  submission robust. Add a CI hook that loads weights on a clean checkout
  so we catch breakage before submitting.
- Card-ID feature hashing in `train/features.py` uses 8 buckets — once we
  start using `all_card_data()` at training time, switch to a real
  card-ID embedding indexed by the master CSV.
- `NUMBER (type=0)` selects (e.g. for "draw N cards" prompts) are routed
  through the same policy; we should special-case them once we see one
  in the wild during a non-mirror match.
- Verify submission deck is competition-legal under the latest expansion
  list (check `kaggle_data/EN_Card_Data.csv` for which card IDs are in
  the legal pool).
