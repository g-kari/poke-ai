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

## LB matchup aggregate after 15 episodes (2026-06-18)

Pulled the replays of every PUBLIC episode our submissions have
played (2-MLP + 3-MLP combined) and classified opponents by deck
archetype using card-ID signatures (e.g., 344+345 = Crustle,
677+678 = Mega Lucario, 119+120+121 = Dragapult ex).

  Archetype          LB record   vs lab bench (80 games)
  Crustle Wall:      0W-3L  (0%)    31.2%  ← lab vastly overestimates
  Mega Lucario:      1W-1L (50%)    36.2%  (in line)
  Dragapult ex:      1W-2L (33%)    20.0%  (in line)
  Casual (other):    6W-1L (86%)    n/a    (not in our bench set)
  total:             8W-7L (53.3%)

Our locally-vendored Crustle Wall rule-based agent (harukiharada's
template) is not a faithful representation of the LB Crustle players.
The 3 LB Crustle opponents (AM, sbite0138, PavelLiashkov) won all
3 games. Either they run tuned variants or our lab agent isn't
playing the wall pattern correctly.

Dragapult and Mega Lucario LB record tracks lab. Casual opponents
are our cushion (86% win rate).

So our LB rating (676.2) is the right number given the matchmaking
mix: ~3 Crustle games at 0%, ~3 Lucario/Dragapult at ~40%,
~9 casual at ~86%. Weighted average ~50%, which is where we are.

The improvement path here is matchup-specific:
  - Crustle Wall counter: would need true LB Crustle replays to train
    against, not the lab template
  - Iono mystery: still no LB Iono opponent seen, can't validate the
    6-17% lab number

LB scores (10 episodes for 3-MLP, 6 for 2-MLP):
  53778627 (3-MLP):  600 -> 695 -> 697 -> 676.2  (5W/5L)
  53776818 (2-MLP):  600 -> 732 -> 586 -> 620.7  (3W/3L)

## Re-tested rejected seeds in 4-MLP context (2026-06-18)

Applied the seed=42 lesson — don't reject on solo alone. Re-benched
the most-promising rejected seeds in 4-MLP context at 80 games per
opp, focusing on whether any of them lifts Iono (our weakest matchup).

  Opponent          3-MLP    4-MLP+seed=8  4-MLP+seed=300  4-MLP+seed=200
  Mega Lucario:     36.2%    26.2%         18.8%           17.5%
  Dragapult:        20.0%    20.0%         22.5%           22.5%
  Iono:             17.5%    6.2%          15.0%           6.2%
  Mega Abomasnow:   31.2%    23.8%         28.7%           25.0%
  Crustle Wall:     31.2%    27.5%         40.0%           30.0%
  overall:          27.3%    20.8%         25.0%           20.2%

All three additions REGRESS overall, and crucially NONE improves
Iono — most actually hurt it. The pattern:

  - seed=300 lifts Crustle (+8.8pp) but tanks Mega Lucario (-17.4pp)
  - seed=200 is a strict downgrade everywhere
  - seed=8 has same issue as seed=300 but milder

No Iono specialist exists in our candidate pool. Iono's 17.5% in the
3-MLP is the floor — every additional member drags Iono toward its
own solo Iono number (6-10%), and the 3-MLP's averaging already
extracted the best possible Iono signal from these three.

Pattern explains why most candidates fail in ensemble: a 4th member
helps one matchup but adds a vote that pulls weak matchups (Iono,
Mega Lucario) down. The 3-MLP's specific Mega Lucario boost (+20pp
from seed=42 averaging) is fragile — any 4th member with a different
Mega Lucario opinion erodes it.

Iono is genuinely architecturally hard: Lightning + Wattrel chains
match against our card features poorly. Probably requires either
(a) deck change away from the Pokemon family we're using or (b) a
richer feature representation that distinguishes Wattrel-chain
states.

Submission stays 3-MLP at LB 697.7.

## seed=42 is the hidden Mega Lucario / Iono specialist (2026-06-18)

Completed the per-member solo bench at 80 games per opp. seed=42 looks
weakest on paper but is the key ensemble member.

  Opponent          seed=20260628  seed=42  seed=100  2-MLP w/o seed=42  3-MLP
  Mega Lucario:     25.0%          18.8%    25.0%     16.2%              36.2%
  Dragapult:        25.0%          20.0%    16.2%     31.2%              20.0%
  Iono:             7.5%           6.2%     10.0%     6.2%               17.5%
  Mega Abomasnow:   27.5%          22.5%    31.2%     23.8%              31.2%
  Crustle Wall:     28.7%          20.0%    38.8%     38.8%              31.2%
  overall:          22.8%          17.5%    24.2%     23.2%              27.3%

seed=42 is the weakest single member (17.5% overall, last in every
matchup). But removing it from the 3-MLP drops the overall ensemble
from 27.3% to 23.2% — a -4.1pp regression.

Per-opp impact of seed=42's presence in the ensemble:
  Mega Lucario:  +20pp  (16.2 -> 36.2 with seed=42)
  Iono:          +11.3pp (6.2 -> 17.5)
  Mega Abomasnow:+7.4pp  (23.8 -> 31.2)
  Dragapult:    -11.2pp (31.2 -> 20.0)  ← seed=42 hurts this one
  Crustle:       -7.6pp  (38.8 -> 31.2)  ← also hurts

So seed=42's ensemble role: lift the matchups where the other two
members are similarly mediocre (Mega Lucario, Iono, Abomasnow). It
drags down on the matchups where the other two are strong specialists
(Dragapult on seed=20260628, Crustle on seed=100). Net +4.1pp.

This is the classic "weak member doing valuable averaging" effect.
Solo benches are misleading — they don't predict ensemble role.

Complete specialty map:
  seed=20260628: Dragapult solo specialist (25.0%)
  seed=42:       Mega Lucario / Iono ensemble specialist (lifts via averaging)
  seed=100:      Crustle solo specialist (38.8%)
  shared gap:    Iono (all three score 6-10% solo, 17.5% ensemble)

The only matchup without a specialist member is Iono. A genuine Iono
specialist would need ≥25% solo on Iono, which none of our trained
seeds hit. May be deck-architectural — Iono uses heavy Lightning
energy and Wattrel chains that our card-pool features can't represent
as well.

Lesson: never solo-bench a seed and reject based on overall solo
strength. Test in the actual ensemble context. seed=42 would have
been rejected if solo-tested today (17.5% << seed=100's 24.2%) but
it's actually critical.

## Solo @ 80 games: members have complementary specialties (2026-06-18)

Until now we'd never benched each ensemble member solo at the tight
80-game-per-opp setting. Doing that reveals real per-seed specialties
that explain how the ensemble lifts overall by +4.5pp.

Per-opp solo bench @ 80 games (400 total):

  Opponent          seed=20260628    seed=100    3-MLP ensemble
  Mega Lucario:     25.0%            25.0%       36.2%   (+11pp from ensemble)
  Dragapult:        25.0%            16.2%       20.0%   (ensemble drags 25%->20%)
  Iono:             7.5%             10.0%       17.5%   (+9pp from ensemble)
  Mega Abomasnow:   27.5%            31.2%       31.2%   (~match)
  Crustle Wall:     28.7%            38.8%       31.2%   (ensemble drags 38.8%->31.2%)
  overall:          22.8%            24.2%       27.3%   (+4.5pp ensemble lift)

Notable specialties:
  - seed=20260628 owns Dragapult (25.0% vs seed=100's 16.2%)
  - seed=100 owns Crustle (38.8% vs seed=20260628's 28.7%)
  - both join forces on Mega Lucario, where averaging lifts both to 36%
  - both struggle on Iono, but averaging still adds +9pp

So the ensemble does its job — it dampens each member's individual
weakness — but at the cost of capping each member's specialty
strength. seed=100's 38.8% on Crustle gets averaged down to 31.2%
because the other members vote differently there.

This suggests weighted / opponent-conditioned averaging could help:
detect Crustle and weight seed=100 higher. But that requires runtime
opponent detection, which is hard from the obs (opp's hand is hidden).
Probably not worth the effort vs other directions.

Submission stays at 3-MLP — the unweighted average is the best we can
do without per-opp detection.

## Tight bench (80 games/opp) reveals seed=8 was noise (2026-06-18)

Took the noise-floor commit's prescription seriously and re-benched
the two candidates at 80 games per opp (400 games total) instead of
the default 30 (150 total). Wilson CI tightens from ~±15pp to ~±9pp.

  3-MLP base:        109-291 (27.3%) Wilson [23%, 32%]
  4-MLP +seed=8:     83-317  (20.8%) Wilson [17%, 25%]

The CIs barely overlap, so the 4-MLP variant is statistically worse
at the 95% confidence level. Per-opponent breakdown:

  Opponent          3-MLP    4-MLP+seed8   Δ
  Mega Lucario:     36.2%    26.2%         -10pp
  Dragapult:        20.0%    20.0%          0pp
  Iono:             17.5%    6.2%          -11.3pp
  Mega Abomasnow:   31.2%    23.8%         -7.4pp
  Crustle Wall:     31.2%    27.5%         -3.7pp

The headline finding: seed=8's "Crustle specialist +10pp" property
that we saw at 30 games per opp was ENTIRELY NOISE. At 80 games per
opp seed=8 actually performs slightly WORSE on Crustle (-3.7pp). It's
not a specialist; it's just a noisier member.

So the previous commits that documented seed=8 as a Crustle counter
were misleading. Updating this finding inline rather than rewriting
those — the path of attempted experiments is itself useful.

True 3-MLP strength on lab: 27.3% across 5 meta agents at 80 games
per opp. The 22-27% range from 30-game runs was just bench noise;
27% is closer to the truth.

Recommendation going forward: 80 games per opp is the new minimum
when evaluating ensemble candidates. The cost (~3 min per bench) is
trivial compared to wasting a daily submission slot on a noise-level
improvement.

## bench_meta noise floor characterized (2026-06-18)

Repeated bench_meta runs of the unchanged 3-MLP ensemble produced
different numbers across days:

  3-MLP run A: 32-88  (26.7%)
  3-MLP run B: 33-117 (22.0%)
  3-MLP run C: 34-116 (22.7%)

Range 22.0-26.7%, ~5pp. With each run still at 30 games per opp /
150 games total. So the 50% solo + bench_meta filter sees only
ensemble changes >5pp as signal; everything smaller is variance.

Recent candidate ensembles, recapped against this noise:

  3-MLP base:                      22-27% (1-3 runs each)
  4-MLP +seed=300 @ 3000ep:        25.8% (Δ within noise)
  4-MLP +wide arch:                18.0% (Δ ~-7pp, below noise floor)
  4-MLP +seed=8 @ lr=8e-4:         20.7% (Δ within noise)
  3-MLP' (seed=100 -> seed=8):     19.3% (Δ ~-5pp, edge of noise)

Only the wider-arch attempt is clearly outside the band, and it's
clearly worse. The seed candidates all land in noise. So the 3-MLP
default really is the local optimum at this evaluation precision.

To distinguish candidates that look marginally better, we'd need
either:
  - much bigger bench (~150 games per opp = 750 total)
  - or paired comparison protocol (same env seeds for both candidates,
    not just bench-level seed=0)
  - or measure on LB (cost: a daily submission slot per attempt)

Saving the 2 remaining daily submission slots since none of the
variants is confident enough to burn one on. Keep 3-MLP as default.

## seed=8 @ lr=8e-4 / 2500ep — rejected, but Crustle specialist (2026-06-18)

Lowered the lr (1e-3 -> 8e-4) and bumped episodes (2000 -> 2500) for
slightly slower / longer training. seed=8 standard architecture.

Solo bench (40 games):
  seed=8 vs linear:        21-19 (52.5%)  ← passes solo
  seed=8 vs random:        40-0  (100%)
  seed=8 vs rule_based:    9-31  (22.5%)  ← best single-MLP solo we've seen
                                            against rule_based

4-MLP ensemble (3-MLP + seed=8) bench_meta (150 games):

                       3-MLP    4-MLP(+seed8)  Δ
  Mega Lucario:        26.7%    26.7%          0pp
  Dragapult:           20.0%    13.3%          -6.7pp
  Iono:                 6.7%     6.7%          0pp
  Mega Abomasnow:      23.3%    13.3%          -10pp
  Crustle Wall:        33.3%    43.3%          +10pp
  overall:             22.0%    20.7%          -1.3pp

Interesting trade-off: seed=8 lifts Crustle by 10pp (the archetype
that beat us on LB) but drags Dragapult and Abomasnow down by
similar amounts. Net -1.3pp.

Given how variance-laden 30-game bench_meta is (we've seen the same
3-MLP land at 18% / 22% / 26.7% across runs), -1.3pp could be noise.
But the cost-benefit isn't compelling enough to swap in.

Discarded. The seed=8 file stays in /tmp/ and the metrics JSON is
committed. Worth coming back to if we want to build a Crustle-focused
variant ensemble for a separate submission.

## Wider MLP arch (128/64/32 + 64/32) — rejected (2026-06-18)

Hypothesis: the standard 64/32 + 32 MLP might be too small to learn
robust play patterns; a wider policy head (128, 64, 32) with a
wider value head (64, 32) could absorb more state-feature
combinations. Trained at lr=1e-3 seed=7 for 2000ep.

Added two CLI args to train.mlp_train for the experiment:
  --hidden-pi  comma-separated policy widths
  --hidden-v   comma-separated value widths

Solo bench (40 games):
  wide vs linear:        21-19 (52.5%)  ← passes solo threshold
  wide vs random:        40-0  (100%)   ← actually perfect, narrow MLPs hit 95%
  wide vs rule_based:    5-35  (12.5%)

The 100% vs random is encouraging — the wider head packs more
distinctions into the engine-prior-augmented logits. Tried it both
solo on bench_meta and as a 4th member of the heterogeneous ensemble:

  wide solo (150 games):           25-125 (16.7%) overall
  3-MLP narrow (baseline):         33-117 (22.0%) overall
  4-MLP heterogeneous (3 narrow + 1 wide): 27-123 (18.0%)

So the wider arch is WORSE both alone and in the mix. Hypotheses:

  - With 2000ep of self-play we don't have enough samples to populate
    the larger parameter space. The optimizer might be settling on
    a local minimum that wins one specific archetype (Mega Abomasnow
    went 26.7% in the ensemble vs 23.3% for 3-MLP — Crustle gained
    too) at the cost of others (Iono dropped from 6.7% to 3.3%).
  - Different MLP widths produce different logit scales. Averaging
    a wide and narrow set in EnsemblePolicy.logits gives the wide
    member disproportionate influence on options where its logits
    are sharper, blurring the consensus we wanted.

50% vs linear is necessary but NOT sufficient. The cheap solo bench
is a noisy filter for outright-bad seeds, not for ranking borderline
acceptable candidates. The comprehensive bench_meta is the truth.

Wider MLP discarded. /tmp/wide_demoted.pt has the file. metrics
JSON kept as the experiment record. The CLI args stay in
train.mlp_train.py as infrastructure for future architecture probes.

## Adding Crustle Wall to bench_meta (2026-06-18)

The LB replay analysis flagged AM (LOSS) as a Crustle Wall archetype
that wasn't in our local opponent set. Pulled
harukiharada/crustle-wall-mirror-ok via the kaggle CLI and added it
as the 5th rule-based opponent.

Per-opponent 30-game bench against 3-MLP:

  vs Mega Lucario:    8-22 (26.7%)
  vs Dragapult ex:    6-24 (20.0%)
  vs Iono's:          2-28 (6.7%)
  vs Mega Abomasnow:  7-23 (23.3%)
  vs Crustle Wall:    10-20 (33.3%)  ← actually our second-best!
  overall (150 games): 33-117 (22.0%)

Surprise: Crustle Wall is one of our easier match-ups in lab. The
single-LB-episode AM loss was just a noisy sample, not evidence that
Crustle counters our deck. (The AM submission may also be a tuned
variant we haven't reproduced exactly.)

Comparing 4-opp vs 5-opp benches at the same sample size:
  4-opp (30 games each, 120 total): 26.7% overall
  5-opp (30 games each, 150 total): 22.0% overall

The per-opp numbers between runs shifted by 6-17pp without any policy
change, so the "26.7% > 22.5%" gap I'd been treating as the 3-MLP
improvement signal was only marginally above noise. The LB
+109pt gap remains the real signal — lab benches are too small to
distinguish marginal gains.

scripts/bench_meta.py now runs the 5-opp version by default.
scripts/rule_based_crustle.py is vendored with attribution (Apache 2.0
inherited from the source kernel). deck_crustle.csv has the 60-card
list. pyproject.toml excludes the new rule-based file from ruff.

## LB replay analysis (2026-06-17 / 18)

Downloaded the 5 PUBLIC episode replays of the 3-MLP submission
(53778627) and identified each opponent's deck from the
`steps[0][0].visualize[0].action` field (which holds both players'
60-card submissions).

  Episode 80323200 / yu_gotou       / WIN / 27 steps  / (early KO)
  Episode 80323539 / Ryota Matsuki  / WIN / 119 steps / (long mid-game)
  Episode 80324055 / Hale Obernolte / WIN / 90 steps
  Episode 80324410 / AM             / LOSS / 62 steps
  Episode 80324949 / YT             / LOSS / 54 steps

The two losses were the rating-relevant data points:

  AM deck   : 344 Dwebble x4 + 345 Crustle x4 + 1147 Jumbo Ice Cream x4
              + 1212 Cook x4 + heal-spam trainers + special energies.
              "Crustle Wall" archetype (matches harukiharada's public
              kernel). Crustle has Stage 1 150 HP with the ability
              "Prevent all damage done to this Pokémon by attacks
              from your opponent's Pokémon ex." Our deck has no ex
              Pokémon so the wall doesn't directly fire, but the heal
              spam (Jumbo Ice Cream heals 80, Cook heals 70) outpaces
              our damage output.

  YT deck   : 119 Dreepy x4 + 120 Drakloak x4 + 121 Dragapult ex x3
              + standard Kiyota trainer suite (1086 Buddy-Buddy Poffin,
              1121 Ultra Ball, 1182 Boss Orders, 1198 Crispin, 1227
              Lillie's). Essentially the Kiyota Dragapult ex template.

So 2/5 LB opponents matched our local meta-bench. We won the other
3 — implying that the LB matchmaking is bi-modal: meta archetype
opponents we struggle with, vs casual / under-developed opponents
we beat. The realized win rate on LB (60%) is way better than the
local meta bench (26.7%) because the LB pool is more varied than
4 hand-tuned rule-based agents.

Implication for next training: the Crustle Wall archetype is not in
our local opponent set. If we expand the multi-opp training to
include this archetype the policy might learn to break the heal
spam (e.g., damage faster, target benched Crustle).

LB tracking (live):
  53778627 (3-MLP):  600 -> 695.5 over 5 PUBLIC episodes (3W/2L)
                     matchmaking has slowed since 14:42 (typical
                     TrueSkill σ shrink after rating stabilizes)
  53776818 (2-MLP):  600 -> 732 -> 633 -> 586.2 over ~6 episodes
                     (now ~110pts behind 3-MLP)

So the +4.2pp lab gap between 2-MLP and 3-MLP translated to a
sustained ~100pt LB gap. Lab signal IS predictive, just compressed.

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
  seed=500:                      42.5% vs linear   ← rejected
  seed=300 @ 3000ep:             50.0% vs linear   ← rejected (borderline)

So 1 of 4 new candidates passed after the initial pair. The pass rate
is lower than 50/50; either random self-play training really is that
sensitive to seed, or the 50% threshold is too tight (Wilson CI at
40 games is roughly ±15pp, so 47.5% rejection is within noise of
50%). Either way, the filter is doing its job — every rejected seed
matched the seed=1024 failure pattern (sub-50% solo, would drag the
ensemble down by 5-15pp).

Future seed candidates probably need either (a) more training
episodes per seed (3000-5000) so the noisy 40-game bench has a
cleaner signal, or (b) a different bench protocol (e.g., 80 games
solo) for tighter CIs. The bench is a noisy 40-game proxy
but it's cheap to run and matches our "submission-budget" risk
appetite — we'd rather hold a verified-strong ensemble than gamble
slots on an untested candidate.

Discarded seeds are stored in /tmp/ (not in train/) so the ensemble
glob loader doesn't pick them up. The metrics JSON is committed so
the failure record is preserved.

**Tested hypothesis (a) — longer training**: trained seed=300 at 3000ep
(50% more episodes than the usual 2000ep). It landed at exactly 50.0%
solo vs linear, right on the threshold. Added it to the ensemble for a
4-MLP bench_meta test anyway:

  3-MLP overall: 26.7% (Lucario 20% / Drag 33% / Iono 13% / Abom 40%)
  4-MLP overall: 25.8% (Lucario 17% / Drag 23% / Iono 17% / Abom 47%)

Net -0.9pp; the borderline candidate did not help. The 50% solo cut
is correctly calibrated: borderline cases drag at least one matchup
even if they help another. seed=300 also rejected. Conclusion:
longer training doesn't fix the seed-quality issue; the random init
really is the dominant factor.

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

## Day-1 #1 Crustle Dashimaki bench (2026-06-18)

LB の Crustle Wall プレイヤー (AM / sbite0138 / PavelLiashkov) は
dashimaki360 が公開した「Beating the Day-1 #1 Crustle bot」notebook の
chip 派生を回している事を確認。これまで対戦相手として vendor していた
harukiharada Crustle Wall とはデッキ構成も判断ロジックも別物。

dashimaki デッキ (60 枚) を `deck_crustle_dashimaki.csv` に展開、agent を
`scripts/rule_based_crustle_dashimaki.py` に vendor。`bench_meta.py` に
6 番目の opponent として追加した。

3-MLP @ 80 games (2026-06-18 計測):

  vs Mega Lucario:        20-60 (25.0%)
  vs Dragapult ex:        16-64 (20.0%)
  vs Iono's:               9-71 (11.2%)
  vs Mega Abomasnow:      17-63 (21.2%)
  vs Crustle Wall (haru): 31-49 (38.8%)
  vs Crustle Dashimaki:   19-61 (23.8%)  ← 新追加
  overall:               112-368 (23.3%)

考察:
- dashimaki Crustle は harukiharada Crustle より **-15pp 強い** (38.8 → 23.8%)
  → 後者は LB Crustle player の代理として **過大評価** していた
- LB 0W-3L vs Crustle Wall は Wilson 95% upper CI ≈ 56%。lab 23.8% は
  この CI 内なので「LB は変則的に弱い」のではなく「サンプル 3 試合の
  通常分散内」。0% は引きが悪かっただけと解釈
- 一方で dashimaki でも 23.8% は依然 overall (23.3%) 程度しかなく、
  Crustle 対策の本命課題は変わらない (active ATK ロックの
  「ふしぎなロックイン」抜けの手段が足りない)
- Iono は 11.2%。前回 30 ゲーム計測で 17.5% だった事と整合せず、
  ベンチノイズが顕在化。80 ゲームでも ±9pp 残るのを再確認

## seed=7 を archive 行きにした件 (2026-06-18)

seed=7 で 2000ep の MLP を学習し、3-MLP に追加して 4-MLP として
Crustle Dashimaki に対する性能を 80 games で測定:

  3-MLP @ 80g vs Crustle Dashimaki: 19-61-0 (23.8%)
  4-MLP @ 80g vs Crustle Dashimaki:  5-75-0 ( 6.2%)  ← -17.6pp

solo bench は取っていないが、ensemble に入れた途端 Crustle Dashimaki
matchup を **17.6pp** 引き落とすロジット平均を出している。これは
seed=42 のような「solo は弱いが ensemble では補完」の逆パターン
（"solo はそこそこだが ensemble に入ると毒"）に該当。

40 games の sweep では:
  vs Crustle Dashimaki: 4-MLP 27.5% / 3-MLP 23.8% で +3.7pp に見えた
これは ±15pp の noise floor 内で誤誘導された数値。**candidate 評価は
candidate を含む ensemble の組み合わせで 80 games 以上必須** をあらためて
確認 (#noise-floor)。

対応:
- `train/mlp_policy_seed7.pt` を `train/archive/` に退避
  (main.py の glob `train/mlp_policy*.pt` は浅いので拾わなくなる)
- 学習履歴は `train/metrics_mlp_seed7.json` に残して再現可能性を確保
- 一旦 3-MLP submission のまま維持

## Crustle-targeted self-play 試み (2026-06-18)

`train/mlp_train.py` に `--opponent` 引数を追加し、mirror self-play では
なく rule_based_crustle_dashimaki を相手にして学習できるようにした
(opponent 側は trace を取らず policy のみ REINFORCE で更新)。

seed=11 で 2000ep 学習 (warm-start なし):

  ep  200: 1.0%
  ep 1000: 2.2%
  ep 2000: 6.0% (recent 200ep)
  → solo 学習中の vs Crustle Dashimaki 勝率

これを 4-MLP (3-MLP + seed11_crustle) として ensemble に入れた結果:

  4-MLP @ 80g vs Crustle Dashimaki: 8-72-0 (10.0%)  ← 3-MLP は 23.8%
  → -13.8pp の劣化

教訓 (seed=7 のときに続く 2 件目の事例):
- **未熟な policy** (random init → 2000ep で solo 6%) のロジット平均を
  ensemble に混ぜると、平均が引き下がる方向に作用する
- 「Crustle 専用学習」自体は方向性として妥当だが、ensemble member として
  通用するには **solo で 20% は超える** ことが必要条件 (seed=100 が 38.8%、
  seed=2 が ~30% 想定 — 3-MLP の Crustle 平均値が 23.8% に乗る)
- ensemble は弱いメンバーを「補完」しない。平均が真理。

対応:
- `train/mlp_policy_seed11_crustle.pt` を archive に退避
- 3-MLP submission のまま維持
- `--opponent` インフラは残す (次回 warm-start 込みで再挑戦する可能性)

次の打ち手 (Crustle 対策):
1. **warm-start 込みの targeted training**: 既存 mlp_policy.pt から
   start して `--opponent rule_based_crustle_dashimaki` で fine-tune。
   solo 20% を超えたら ensemble に入れる
2. **active ATK ロック (ふしぎなロックイン) を抜く特徴量**: 場のポケモンが
   ex かどうか、bench に non-ex がいるか等を feature 化
3. **PIMC の rollout policy として既存 ensemble を使う** (NOTES の打ち手 #1)

### 上記 #1 の追試 (2026-06-18)

seed=100 (3-MLP の Crustle 友達, solo 15%) を warm-start に置いて、
lr=3e-4 で 1000ep の Crustle dashimaki 専用 fine-tune を回した。

  ep  100: 14%
  ep  500: 7%
  ep  800: 14%
  ep 1000: 7%   ← recent 100ep 勝率 (mirror selfplay と単純比較不可)
  solo @ 40g post-ft: 12.5%  (元 15.0%、ノイズ内、改善なし)

考察:
- lr=3e-4 でも catastrophic forgetting の影響あり (元 15% → 12.5%)
- recent 勝率が振動 (7→14→7) で安定収束していない
- 単一相手の rule-based に対する REINFORCE は **reward 分散が極大** で
  policy gradient が安定化しないと思われる
- 4-MLP として ensemble bench は割愛 (solo 12.5% は 3-MLP 平均 23.8% を
  引き下げる方向で確定)

次の改善案:
- **mixed-mode opponent**: 50% mirror selfplay + 50% Crustle、reward 分散を
  小さく保ちつつ Crustle 経験を増やす
- **lr=1e-4 以下** でさらに低く設定し、Crustle gradient を小さい摂動として
  既存 policy に加える
- **value baseline** が tanh(V(s)) を返す現在の実装が、極端な hard matchup
  (Crustle dashimaki, win rate 6-15%) では advantage 計算を崩している可能性。
  reward を 0/1 scale で見直す

### 上記 #1 mixed-mode 追試 (2026-06-18)

`run_episode` に `opponent_prob` 引数を追加し、エピソード単位で確率的に
mirror selfplay / 対 Crustle dashimaki を切替えるように改造
(`--opponent-prob 0.5`)。lr=1e-4 まで下げて 1000ep warm-start fine-tune。

  ep  100 recent: 0.43 (mirror 含むため当然高い)
  ep  500 recent: 0.37
  ep 1000 recent: 0.29
  solo @ 40g post-mix-ft vs Crustle Dashimaki: **5.0% (元 15.0%)**

**さらに悪化。mixed-mode は Crustle 改善を生まなかった。**

なぜ:
- mirror selfplay は recent ~50% の高勝率 → gradient signal が大きい
- Crustle 対戦は recent ~5-15% の低勝率 → gradient signal が薄い
- 結果として policy gradient は mirror に最適化される方向に偏り、Crustle
  特性 (warm-start 由来 15%) を**早く忘れる**

pure ft (lr=3e-4 100% Crustle): 15→12.5%
mixed ft (lr=1e-4 50/50)       : 15→5%
→ **どちらの方向でも fine-tune で Crustle solo 性能を上げられず**

総括: 単純な REINFORCE + 学習相手切替えでは Crustle dashimaki への
特化は実現できない。理由として推定:
1. value baseline tanh(V(s)) は ±1 で頭打ち、5-15% 勝率の hard matchup
   では advantage がほぼ常に負 → policy gradient が全行動を一様に押し下げ
2. fight-or-flight (=長引かせる) の Crustle 戦略は探索的に発見できる
   水準を超えており、random init からの探索では到達しない局所最適

次サイクル候補 (方向性転換):
- **Crustle 検出 + 専用 agent 切替**: obs から相手 active が Crustle と
  特定できれば、main.agent 内で rule-based のような heuristic に切替える。
  Crustle の active card ID を identify するルートを `cg/api.py` から探す
- **value baseline の reward 正規化**: 0/1 scale + EMA で baseline を作り、
  hard matchup でも advantage が機能するように
- **PIMC でのルックアヘッド**: Crustle のような hp 大量の defensive deck は
  search で「TKO までのターン数」を読めば 1-2-ply でも判断改善できる

### Iono 専用 ft が初めて solo 改善 (2026-06-18)

Iono は 3-MLP @ 80g で 11.2% と最大の弱点。seed=0 (base mlp_policy.pt) を
warm-start に置き、Iono を opponent に pure mode (opp-prob=1.0)、
lr=3e-4 で 1000ep fine-tune。

  seed=0 base solo  @ 40g vs Iono: 2-38-0 ( 5.0%)
  seed=0 iono_ft    @ 40g vs Iono: 4-36-0 (10.0%)  ← **+5.0pp**

Crustle ft は全て solo を悪化させた (15→12.5%、15→5%) が、Iono は
solo で初めて改善。仮説: Iono は「打点を出してくる active deck」なので
reward 分散が Crustle ほど偏らず、policy gradient が機能する。

ただし 40g なので noise floor ±15pp 内。本判定には 80g 必須。

5-MLP (3-MLP + iono_ft + duplicate) のクイック bench (40g):
  vs Iono            : 17.5% (3-MLP 11.2% → +6.3pp)
  vs Crustle Dashimaki: 12.5% (3-MLP 23.8% → -11.3pp)

トレードオフ顕在化:
- Iono 改善 +6.3pp
- Crustle Dashimaki 悪化 -11.3pp
- net 効果は overall 80g bench でないと判定不可

次サイクル: 重複なし 4-MLP (3-MLP + seed0_iono_ft) で全 6 opp @ 80g
bench を取り、overall 改善があれば LB submit する。
現在 iono_ft policy は `train/archive/mlp_policy_seed0_iono_ft.pt` に
退避済み (次サイクルで `train/` に戻して 4-MLP 化する)。

### 4-MLP (3-MLP + Iono ft) 80g 結果 — 採用見送り (2026-06-18)

重複なしで `train/` に戻し、全 6 opp @ 80g で本判定:

| matchup | 3-MLP @ 80g | 4-MLP @ 80g | delta |
|---|---|---|---|
| Mega Lucario       | 25.0% | 22.5% | -2.5pp |
| Dragapult ex       | 20.0% | 15.0% | -5.0pp |
| **Iono's**         | 11.2% |  8.8% | **-2.4pp** ← 期待は +6pp、80g で逆転 |
| Mega Abomasnow     | 21.2% | 28.7% | +7.5pp |
| Crustle Wall (haru)| 38.8% | 28.7% | -10.1pp |
| Crustle Dashimaki  | 23.8% |  6.2% | **-17.6pp** |
| **overall**        | **23.3%** | **18.3%** | **-5.0pp** |

つまり 40g クイック結果 (Iono +6.3pp / Crustle -11.3pp) は両方向で誤り:
80g 真実は Iono **-2.4pp** / Crustle Dashimaki **-17.6pp**。

判定: LB submit せず、`train/archive/` 戻し。

**今サイクルの教訓 (= seed=7 サイクルから 2 回目の再演)**:
- **40g クイック bench は完全に信頼できない**。±15pp の noise が
  実質的に「逆方向の結果」を返す
- **新候補メンバーは ALWAYS 80g full bench から始める** (40g スキップ)
- solo bench の改善 (Iono: 5→10%) も ensemble 平均では別物。
  **solo positive は ensemble positive を保証しない**

## 失敗試行サマリ (2026-06-18)

| 試行 | 学習設定 | solo @ 40g | 4-MLP @ 80g vs 該当 opp | 採用? |
|---|---|---|---|---|
| seed=7 | random 2000ep mirror | - | Crustle Dashi -17.6pp | No |
| seed=11_crustle | random 2000ep pure | - | Crustle Dashi -13.8pp | No |
| seed=100_crustle_ft | warm-start 1000ep pure lr=3e-4 | 12.5% (元 15%) | - | No |
| seed=100_crustle_mix | warm-start 1000ep mixed lr=1e-4 | 5.0% (元 15%) | - | No |
| seed=0_iono_ft | warm-start 1000ep pure lr=3e-4 | 10.0% (元 5%) ← solo + | overall -5.0pp | No |

5 連敗。**シングル MLP ensemble に新メンバーを追加するアプローチは行き詰まり**。
方向転換が必要 (Crustle 検出切替 / value baseline 正規化 / PIMC / デッキ変更)。

## 衝撃の発見: rule-based(Mega Lucario) は overall 46.5% (2026-06-18)

`scripts/bench_rule_based_lucario.py` を新規作成し、Kiyota の rule-based
Mega Lucario agent (`scripts/rule_based_agent.py`) を主役にして 6 opp @ 80g
で測定。

| matchup | rule_based(Lucario) | 3-MLP (我々の LB 666.3) | delta |
|---|---|---|---|
| Mega Lucario (mirror) | 46.2% | 25.0% | +21.2pp |
| Dragapult ex          | 53.8% | 20.0% | +33.8pp |
| **Iono's**            | **83.8%** | 11.2% | **+72.6pp** |
| Mega Abomasnow        | 47.5% | 21.2% | +26.3pp |
| Crustle Wall (haru)   | 25.0% | 38.8% | -13.8pp |
| Crustle Dashimaki     | 22.5% | 23.8% | -1.3pp (互角) |
| **overall**           | **46.5%** | 23.3% | **+23.2pp** |

つまり、これまで何十時間も注いだ MLP 学習路線より、**Kiyota 公式 sample
agent + Mega Lucario deck を素で提出する方が約 2 倍強い**。
特に Iono **+72.6pp** は致命的: 我々の MLP は Iono に対して全く機能してない。

### LB submit 切替の準備済み

1. `main_rule_based.py` — Kaggle exec() shape に対応した wrapper
   (`__file__` 罠は前回と同じ `contextlib.suppress(NameError)` 対処)
2. `make_submission_rule_based.sh` — `main_rule_based.py` → `main.py`、
   `deck_mega_lucario.csv` → `deck.csv` にリネームしてバンドルする
3. `scripts/check_main_exec.py --no-policy` — 非 MLP submission の sandbox 検証
4. ローカル build & verify は完了 (`submission_rule_based.tar.gz`)

### 切替 submit する場合のコマンド

```bash
./make_submission_rule_based.sh
.venv/bin/kaggle competitions submit -c pokemon-tcg-ai-battle \
    -f submission_rule_based.tar.gz \
    -m "Switch to rule-based Mega Lucario (lab 46.5% vs 3-MLP 23.3%)"
```

### user 確認待ち事項

- **submit する/しない判断はユーザに委ねる** (LB スコア 666.3 を
  上書きするアクション)
- rule-based(Lucario) の Crustle Wall 25% (- 13.8pp) は注意点。LB 上の
  Crustle 比率が高いと overall 効果が変わる可能性あり (ただし 3-MLP の
  Crustle 38.8% も実 LB プレイヤー (dashimaki 派) では 22.5% 相当)
- ストラテジー部門 (締切 9/14) は別途、MLP 学習の知見は残しておく

## さらに強い候補発見: rule_based(Iono) 64.0% (2026-06-18)

`scripts/bench_rule_based_lucario.py` を generalize して
`--subject {lucario, dragapult, iono, abomasnow, crustle, crustle_dashimaki}`
を切替えられるようにし、各 rule-based agent を主役にして 80g/opp で総当り。

### 候補ランキング @ 80g (2026-06-18 拡張)

| subject | overall | Mega Luc | Drag | Iono | Aboma | Crustle Wall | Crustle Dashi |
|---|---|---|---|---|---|---|---|
| **CrustleDashi** | **67.3%** | 83.8% | **100%** | **11.2%** ★ | 73.8% | 85.0% | 50.0% |
| Iono | 64.0% | 21.2% ★ | 46.2% | 57.5% | 68.8% | **97.5%** | **92.5%** |
| Dragapult | 48.1% | 51.2% | 56.2% | 70.0% | 52.5% | 58.8% | 0.0% ★ |
| Lucario | 46.5% | 46.2% | 53.8% | 83.8% | 47.5% | 25.0% | 22.5% |
| Abomasnow | 40.0% | 31.2% | 46.2% | 28.7% | 50.0% | 67.5% | 16.2% |
| 3-MLP (我々) | 23.3% | 25.0% | 20.0% | 11.2% | 21.2% | 38.8% | 23.8% |

★ = single critical weakness。Iono と CrustleDashi は相互天敵 (Iono は Lucario 21.2% で苦しく、CrustleDashi は Iono 11.2% で苦しい)。

### 補足: deck 探索状況 (2026-06-18 夜)

既存の vendored deck/agent ペアは 6 つ:
deck_mega_lucario, deck_dragapult, deck_iono, deck_abomasnow,
deck_crustle (haru), deck_crustle_dashimaki。

残 subject 未計測: **Crustle Wall (haru)**。次サイクルで bench 予定。
完了したら 6 subject 全網羅。

### 6 subject 全網羅完了 (2026-06-18 後半)

最終ランキング @ 80g (全 6 subject):

| rank | subject | overall | 致命的弱点 |
|---|---|---|---|
| 1 | **CrustleDashi** | **67.3%** | Iono 11.2% |
| 2 | Iono | 64.0% | Mega Lucario 21.2% |
| 3 | Dragapult | 48.1% | Crustle Dashi 0.0% |
| 4 | Lucario | 46.5% | Crustle Wall 25.0%, Crustle Dashi 22.5% |
| 5 | Abomasnow | 40.0% | Iono 28.7%, Crustle Dashi 16.2% |
| 6 | CrustleWall (haru) | 36.9% | Iono 1.2%, Crustle Dashi 10.0% |

- 上位 2 つ (CrustleDashi, Iono) は相互天敵関係
- LB の deck 分布が分からない以上、どちらが優れるかは経験的判断
- 全 6 subject の submission ready (`make_submission_rule_based*.sh`)

### CrustleDashi submission build 完了

`main_rule_based_crustle_dashimaki.py` + `make_submission_rule_based_crustle_dashimaki.sh`
作成、`submission_rule_based_crustle_dashimaki.tar.gz` (1.06MB) を build &
sandbox verify 完了。deck = [344, 344, 344, ...] (Crustle 系 head)。

### Submission 履歴 (2026-06-18)

| ref | file | status | publicScore |
|---|---|---|---|
| 53793417 | submission_rule_based_iono.tar.gz | COMPLETE | **600.0** (初期値、評価少) |
| 53778627 | submission.tar.gz (3-MLP) | COMPLETE | 679.6 |
| 53776818 | submission.tar.gz (2-MLP fix) | COMPLETE | 613.3 |
| 53776705 | submission.tar.gz (2-MLP) | ERROR | - |

**Iono submission の public score 600.0 は TrueSkill 初期値 μ₀=600 から
動いていない**。試合数が少なく評価が始まったばかりの可能性。3-MLP も
最初は 666.3 だったのが 679.6 に動いたので、Iono もこれから動くはず。
60 分後/2 サイクル後にステータス再チェック予定。

### 観察待ち事項

- Iono の LB スコアが μ₀ 周辺から動くか
- 動かない/下がる場合: rule_based(Iono) は実 LB プレイヤーには通用しない可能性 → CrustleDashi 案にスイッチ
- 上がる場合: 64.0% lab signal は LB に translate している

## romanrozen V6 を vendor (2026-06-18 night)

Kaggle public kernel `romanrozen/strong-start-crustle-lucario-agent-v6-lb-860`
(36 votes、author 自身 LB 860+ を主張)。
Lucario + Hariyama + Solrock ハイブリッド deck で、CRUSTLE_AWARE=True により
Crustle 検出時に non-ex Hariyama (Crustle "Mysterious Rock Inn" 抜け)
attack route を取る anti-Crustle 強化版。

- vendored as `scripts/rule_based_romanrozen_v6.py` (env var
  RULE_DECK_PATH_ROMANROZEN_V6 で deck 切替)
- deck saved as `deck_romanrozen_v6.csv` (OLD_DECK = 60 cards)
- USE_SEARCH=False (heuristic only、安全)

### V6 @ 80g bench

  vs Mega Lucario:    35-45 (43.8%)  ← lowest
  vs Dragapult ex:    46-34 (57.5%)
  vs Iono's:          61-19 (76.2%)
  vs Mega Abomasnow:  43-37 (53.8%)
  vs Crustle Wall:    41-39 (51.2%)
  vs Crustle Dashi:   52-28 (65.0%)
  overall:           278-202 (57.9%)

### 7 subject 最新ランキング @ 80g

| rank | subject | overall | min (= 致命弱点) |
|---|---|---|---|
| 1 | CrustleDashi | 67.3% | Iono 11.2% |
| 2 | Iono | 64.0% | Mega Lucario 21.2% |
| 3 | **RomanrozenV6** | **57.9%** | **Mega Lucario 43.8%** (致命弱点なし) |
| 4 | Dragapult | 48.1% | Crustle Dashi 0% |
| 5 | Lucario | 46.5% | Crustle Wall 25%, Crustle Dashi 22.5% |
| 6 | Abomasnow | 40.0% | - |
| 7 | CrustleWall (haru) | 36.9% | Iono 1.2% |

V6 は overall は CrustleDashi/Iono に届かないが、**全 matchup で 43.8% 以上**を
維持する分散の小さい候補。author 自身 LB 860+ を主張するので、実 LB では
peak 値より分散の小ささが効く可能性。

### V6 submission build 完了

`main_rule_based_romanrozen_v6.py` + `make_submission_rule_based_romanrozen_v6.sh`
作成、`submission_rule_based_romanrozen_v6.tar.gz` (~1MB) を build &
sandbox verify 完了。

### Submit 候補 4 つ ready
- submission_rule_based_iono.tar.gz (現在 LB 上、評価進行中)
- submission_rule_based_crustle_dashimaki.tar.gz (lab 67.3%、未 submit)
- submission_rule_based_romanrozen_v6.tar.gz (lab 57.9%、未 submit、author LB 860+)
- submission_rule_based.tar.gz (Lucario, lab 46.5%、未 submit)

## Iono LB 762.2 で大成功 (2026-06-18 night)

53793417 (Iono) public score 推移:
- T+0:   600.0 (TrueSkill μ₀ 初期値)
- T+30m: 615.9 (+15.9、評価開始)
- T+1h:  **762.2** (+146.3、3-MLP 679.6 を **+82.6 で overtake**)

**lab signal は LB に translate する** ことが実証された。
これは大きな結論で、以下が含意される:
- CrustleDashi (lab 67.3%) は更に高得点取れる可能性
- V6 (lab 57.9%, author 主張 LB 860+) も挑戦候補
- 1 日 5 件枠、本日 1 件 (Iono) 投げたので残り 4 件可能

### CrustleDashi も submit 完了 (2026-06-18)

53794617 (CrustleDashi) PENDING、評価待ち。user 選択で「Iono を上回るか比較」。

## メタ情報: **イワパレス (Crustle) がメタ** (2026-06-18 user 知見)

User より「LB ではイワパレス (Crustle) がメタ」情報を共有。
これは戦略的判断を更新する材料:

- **Iono (Crustle 97.5/92.5%) は引き続き優位** ← 既に LB 762.2 で実証
- **CrustleDashi (Iono 11.2%) は LB の Iono プレイヤー比率次第**
  - もし Iono もメタなら CrustleDashi は危険
  - もし Crustle 主導でメタなら CrustleDashi は mirror 50% で平均的
- **V6 (anti-Crustle 強化、Crustle 51.2/65.0%) は安全な選択**
  - 全 matchup 43%+ で分散小さい
  - author 自身 LB 860+ を主張

53794617 の結果が出れば、Iono と CrustleDashi の直接比較が可能。
**Crustle メタ前提なら次の打ち手は V6 が最も安全**。

## kojimar agent も vendor (2026-06-18 night)

`kojimar/validated-rule-based-agent-matchup-tests` (32 votes) を vendor。
Lucario + Hariyama + Solrock 構造。

### kojimar @ 80g
- Mega Lucario 60.0%, Dragapult 61.3%, Iono **81.2%**
- Aboma 62.5%, Crustle Wall 30.0%, Crustle Dashi 21.2%
- overall: **52.7%**

Kojimar は Crustle 対策が弱い (V6 と異なり anti-Crustle 専用ルーチンなし)。
**Crustle メタ環境では V6 より劣る**ので即 submit 候補ではない。

## LB 環境調査 (2026-06-18, user 指摘で着手)

### LB トップ 40 のスコア分布

| rank | team | score | submit date |
|---|---|---|---|
| 1 | onechan1 | **1308.7** | 06-17 |
| 2 | DENPA92 | 1230.4 | 06-17 |
| 3 | Kyo_s_s | 1211.6 | 06-17 |
| 4 | くりたんとたけちゃん | 1179.9 | 06-17 |
| 5 | YT | 1175.8 | 06-17 |
| 6 | tubotu | 1174.6 | 06-18 |
| ... | (1100 帯多数) | ... | |
| 18 | sbite0138 (Crustle派) | 1096.8 | 06-17 |
| 20 | Kuroneko | 1082.3 | 06-17 |
| ~37 | takuyay (dashimaki本人?) | 1047.6 | 06-16 |
| ~38 | AM (Crustle派) | 1069.2 | 06-17 |

### 我々の位置

- Iono 762.2: トップ 40 (1035.4) より **-273 下**
- CrustleDashi 718.0: 更に下
- LB 上位 1100+ には rule-based + meta deck では届かない

### 環境の含意

1. **rule-based agent の天井 = ~1100** (anti-Crustle 強化版でこの水準)
2. **トップ 1200+ は別技術**: PIMC / MCTS / 強化学習 / 洗練された深いルックアヘッド
3. **「LB 860+」を author が主張する V6 ですら、実 LB では更に上の人々が大勢**
4. Crustle 派 (sbite0138, AM, takuyay/dashimaki) は 1047-1097 帯。**Crustle メタ
   は事実だが、最上位ではない**
5. 我々の現状 762.2 は、まず 1000 を超える事が短期目標、次に 1100、最終的に 1200+

### 戦略再考

- 短期 (今日): rule-based 系で 800-900 を目指す (V6 評価結果次第)
- 中期: PIMC 実装 — NOTES の「打ち手 #1」、cg.api.search_begin/step が使える
- 長期 (締切 8/16 まで 2 ヶ月弱): PIMC + 強化学習 hybrid を目指す

## PIMC スモーク成功 (2026-06-18 user 指摘で本格着手)

User 知見「ルールベースより ML、Kaggle がそう」を受けて方針再転換。
PIMC で隠し情報サンプリング → 学習済み policy/value で rollout → Q 評価、
AlphaZero スタイルが LB 1200+ の本命と判断。

### スモーク (`scripts/pimc_smoke.py`)

self-play 中の自 turn で `cg.api.search_begin` を呼ぶ smoke を実装。

第 1 ハマり: `agent_observation=obs` (dict) で渡すと
`AttributeError: 'dict' object has no attribute 'search_begin_input'` 失敗。
docstring は「pass obs as is」と書いてあるが、C 側は属性アクセスする
ので **`to_observation_class(obs)` で dataclass 化が必須**。

修正後の結果:
- 1 turn 目: **PIMC succeed** (sid=1, dt=**0.2ms**)
- 以降: `ValueError: You need to predict the opponent's Active Pokémon.`

判明事項:
1. **PIMC は実機で動く** — vendored cg/api.py の search_begin は機能
2. **超高速 0.2ms/call** — Kaggle の 3 秒 budget なら理論上 15,000 回
   search 可能。性能制約は無視できる
3. **opponent の hand / active を空で渡すと拒否される** ので、
   情報集合サンプリング (= 何らかの予測値) が必須。これが PIMC 実装の核

### 次サイクル予定 (PIMC 実装段階 #2)

1. **opponent_hand / opponent_active サンプリングを実装**:
   - obs から見える: 相手 deck カウント、prize 枚数、discard、手札枚数
   - サンプリング元: 「全カードプールから basic ポケモンを優先抽出」
     または「LB の人気 deck (Iono/Lucario/Crustle) から仮定」
2. **1-ply rollout で Q 評価**:
   - root の各 option について search_begin → search_step → 終局 reward
   - opp_hand を K 回サンプリング、各 option の期待 Q 計算
   - argmax で行動選択
3. **既存 3-MLP / Iono rule-based を rollout policy として再利用**

これが完成すれば、PIMC が初めて「対戦相手によって戦略を変える」
適応的 agent になる (= search で相手の手を読んで反応する)。

### 1-ply PIMC agent 初版 (2026-06-18)

`train/pimc_agent.py` + `scripts/bench_pimc.py` を実装。
- 各 root option について search_begin + search_step(1-ply)
- value heuristic: prize 差分 (`opp_taken - my_taken`)
- opp_hand サンプリング: 相手 deck からランダム
- time budget 1.5s/decision、option cap 8 (root branching 対策)
- 例外時は engine order fallback で crash 防止

bench (10g/opp、 我々のオリジナル deck、Iono を opp deck と仮定):
  vs Mega Lucario:    3-7  (30.0%)
  vs Dragapult ex:    1-9  (10.0%)
  vs Iono's:          1-9  (10.0%)
  vs Mega Abomasnow:  0-10 ( 0.0%)
  vs Crustle Wall:    0-10 ( 0.0%)
  vs Crustle Dashi:   0-10 ( 0.0%)
  overall: 5-55 ( 8.3%) ← 3-MLP 23.3% より悪い

考察:
1. **我々のオリジナル deck の弱さが直接露出** — PIMC を rule-based の
   強 deck (Iono / Crustle Dashi) で動かすと性能上がるはず
2. **1-ply prize_delta は信号不足** — 1 turn で prize 枚数の変化は稀。
   多くの option で score=0 となり、tie-breaker (= engine order) に
   なってしまう
3. **rollout 深さ不足** — 真の PIMC は終局までシミュ。最低でも 3-5 ply
4. **opp_hand サンプリングが poor** — Iono 仮定だが、Mega Lucario 相手で
   Iono hand と仮定すると評価がズレる

改善路線:
- **値関数の強化**: prize 差分だけでなく active HP / bench 数 / 場のエネ
- **multi-ply rollout**: search_step を複数回、または rule-based playout
- **deck 切替**: PIMC を Iono deck で動かす実験 (現状の我々の deck が弱い)
- **rollout policy として MLP/rule-based を使う**

## LB 更新 (2026-06-18)

| ref | file | publicScore | 変化 |
|---|---|---|---|
| 53794617 | CrustleDashi | **811.9** | 718.0 → **+93.9** ← **新ベスト** |
| 53794828 | V6 | **801.9** | 初評価 |
| 53793417 | Iono | 762.2 | 安定 |
| 53778627 | 3-MLP | 679.6 | safe |

CrustleDashi が +93.9 で大躍進、Iono を抜いて新ベスト。
V6 は初評価 801.9 で堅実、author 主張 LB 860+ には届かず。
最新 2 件 tracking: V6 + CrustleDashi (Iono は外れた可能性、ただしスコアは保持)。
LB トップ 40 (1035.4) までまだ -223、PIMC + 学習が必要なギャップ。

## opponent pool 多様化 fine-tune (2026-06-18 user 議論後)

「データの質を上げる」改善 #2 (相手プール多様化) を実装:
`train/mlp_train.py` に `--opponent-pool` (カンマ区切り module 名)
を追加、毎エピソード pool からランダム選択。

学習設定: 5 opp pool [Iono, CrustleDashi, V6, Lucario, Dragapult]、
warm-start mlp_policy.pt、lr=1e-4、1500ep、seed=42。

学習中: recent 勝率 0.18-0.28 (mix opp の平均)、安定。

### solo @ 20g 結果

  vs Mega Lucario:    4-16 (20.0%)
  vs Dragapult ex:    8-12 (40.0%) ← 改善傾向？
  vs Iono:            0-20 ( 0.0%) ← **致命的悪化**
  vs Mega Abomasnow:  4-16 (20.0%)
  vs Crustle Dashi:   0-20 ( 0.0%) ← **致命的悪化**
  vs V6:              4-16 (20.0%)

**判定: pool 多様化だけでは「全方位中庸」policy になり、強い相手 (Iono /
Crustle Dashi) には全敗。**

仮説の検証結果:
- 当初仮説: 「相手によって戦略を変えるシグナルが学べる」
- 実際: **policy は obs から相手 deck を識別する手がかりが無い**
  (features.py は自分の場の情報のみ、相手の active カード ID 等を見ていない)
- 結果として 5 opp を平均的に処理する妥協 policy になり、各 opp 専用の
  戦略は学習されない

教訓:
**pool 多様化は features.py の表現力強化と同時にやらないと効かない**。
具体的には:
- 相手 active カード ID embedding
- 相手 bench 数 / energy 数 / discard pile 主要カード
- 相手 deck pattern 推定 (Iono が Wattrel chain を出してれば Iono と分かる)

これらの features があれば policy は「相手が誰か」を identify でき、
pool 学習で「相手によって戦略を変える」を学習できる。

### archive

- `train/archive/mlp_policy_pool5.pt` (1500ep pool training)
- `train/metrics_mlp_pool5.json` (学習履歴)

### 次サイクル候補

優先順 (前回議論より):
1. **features.py 強化** (相手 active embed、 deck pattern 推定) → これが前提
2. **value baseline 0/1 + EMA** (hard matchup での gradient 改善)
3. PIMC + 学習済み NN value (AlphaZero スタイル、本命)
4. pool training は features 強化後に再試

## features.py 強化 + fresh learning 試行 (2026-06-18)

train/features.py の STATE_DIM を 40 → **60** に拡張。追加 20-d:
- `f[40..55]`: opp の active/bench/discard カード ID を 16 buckets に hash
  (active は weight 2.0、bench は 1.0、discard は 0.5)
- `f[56..59]`: 自分の active カード ID を 4 buckets に hash (mirror sense)

実装は `_accumulate_card_buckets()` で sample-level normalize。
これで policy は「相手の場のカード ID 構成」を fingerprint として認識でき、
理論上「相手が Iono か Crustle か」を識別して戦略を切替えられる。

### fresh learning (pool5、 1500ep、 lr=5e-4) 結果

  vs Mega Lucario: 2-18 (10.0%)
  vs Dragapult ex: 1-19 ( 5.0%)
  vs Iono:         0-20 ( 0.0%)
  vs Mega Aboma:   5-15 (25.0%)
  vs Crustle Dashi:0-20 ( 0.0%)
  vs V6:           3-17 (15.0%)

avg ~9.2%、3-MLP 23.3% よりまだ弱い。 fresh init から features60 + 1500ep
では未収束 (容量増の分、学習量も必要)。

### 学び

- features 拡張は技術的には実装 OK (smoke 通った、normalize 正しく動く)
- ただし fresh init では 1500ep では不足、もっと長い学習 (3000-5000ep) が必要
- STATE_DIM 変更すると warm-start 不可 → ゼロから学習し直し、時間コスト大
- 次サイクル: features60 で 5000ep 学習 + 評価

### archive

- `train/archive/mlp_policy_features60.pt` (1500ep、 fresh、 pool5)
- `train/metrics_mlp_features60.json`

### 🚨 reverted (本サイクル内、submission 保護)

STATE_DIM=60 を維持すると、既存 mlp_policy*.pt (40-d で訓練) が
shape mismatch でロード不可となり、3-MLP submission が壊れる
(`_POLICY is None`)。submission を保護するため:
- `STATE_DIM` を 40 に戻した
- `_accumulate_card_buckets` 関数は残置 (= unused helper、 後で復活)
- 60-d feature ロジック自体はコミット履歴 + `_accumulate_card_buckets`
  に残るので次サイクルで「features60.py を別ファイル化、新 policy 専用」
  すれば再利用可能

**教訓**: STATE_DIM 変更は破壊的、既存 submission を壊す。
作業ブランチで feature 拡張 → 再 train → 評価 → main マージ、 が正しい順序。
1 サイクル内では「features 増 + 学習 + 評価」を atomically 実行できない。

## カードデータ分析 (Task #107 第一歩、 2026-06-18)

`scripts/analyze_cards.py` を新規実装、 `kaggle_data/EN_Card_Data.csv`
(2022 cards) を読んで category 別集計と効率指標を出す。

### カテゴリ別

- Basic Pokémon: 958 / Stage 1: 618 / Stage 2: 229
- Item: 82 / Supporter: 61 / Tool: 28 / Stadium: 26
- Special Energy: 12 / Basic Energy: 8

### 主要発見

**Weakness 分布 (Pokemon HW + Stage1/2 の weakness type 集計):**

| weakness type | count |
|---|---|
| {R} Fire | **361** |
| {F} Fighting | **323** |
| {L} Lightning | 258 |
| {G} Grass | 247 |
| {W} Water | 166 |
| {D} Dark | 157 |
| {M} Metal | 156 |
| {P} Psychic | 66 |

→ **LB 環境では Fire/Fighting attacker が 600+ cards に super-effective**
を持つ。 Cinderace ex (Fire, 280 dmg/1 energy) や Mega Camerupt ex
(280 dmg/1 energy) が「最強候補」 になり得る。

**damage/energy 効率トップ:**

  Mega Camerupt ex (Fire)   Volcanic Meteor   280 dmg / 1 energy
  Cinderace ex     (Fire)   Flare Strike      280 dmg / 1 energy
  Palafin ex       (Water)  Giga Impact       250 dmg / 1 energy
  Crabominable     (Fighting) Haymaker        250 dmg / 1 energy
  Conkeldurr       (Fighting) Gutsy Swing     250 dmg / 1 energy

**HP/(retreat+1) 効率トップ (Basic Pokemon):**

  Mega Latias ex  HP 280 / retreat 1 → 140
  Mega Diancie ex HP 270 / retreat 1 → 135
  Mega Audino ex  HP 270 / retreat 1 → 135
  Mega Hawlucha ex HP 250 / retreat 1 → 125

### 我々の deck.csv の分析

  Basic Energy: 33 (Water 系)
  Supporter: 8 / Item: 5 / Stadium: 2 / Tool: 2
  Basic Pokémon: 6 (Kyogre x2、 Snover x4)
  Stage 1: 4 (Mega Abomasnow ex x4)

  → **我々のデッキは Water 系 Mega Abomasnow ex 軸**
  → {W} weakness は 166 cards で中の下、 LB の Fire メタ {R} 361 を狙えない
  → これが LB 上での agent の苦戦の構造的要因かも

### deck-builder agent への入力候補

将来的に Task #107 deck-builder agent を作る際の評価関数案:
1. **damage efficiency**: 各 Pokemon の最強 attack の dmg/energy
2. **HP efficiency**: HP/(retreat+1) で受けの強さ
3. **type coverage**: deck の weakness をカバーする attacker (mirror 対応)
4. **meta-fit**: deck の主要 attacker が LB の Pokemon weakness 集計を
   どれだけ突けるか (= e.g. Fire 系なら 361 cards に super-effective)

### data/ レイヤー整備 (Task #107 基盤、 2026-06-18)

`scripts/analyze_cards.py --json data/cards.json` で deck-builder の
入力 JSON を生成:

  data/cards.json (401 KB、 1805 Pokemon entries)
    - metadata: total_cards, source
    - categories: 9 カテゴリ別 count
    - weakness_distribution: {R}=361, {F}=323, {L}=258, ...
    - top_hp_efficiency: 30 entries (Mega Latias ex 等)
    - top_damage_efficiency: 30 entries (Mega Camerupt ex 等)
    - pokemon_db: 1805 entries (card_id, name, hp, type, weakness, retreat, ex, mega)

`data/matchups.json` で対戦相性表を構造化:
  - 10 subjects (8 rule-based + 3-MLP + V60 EXT)
  - 各 subject の overall_winrate / critical_weakness
  - matchup_winrates_80g: 10×6 マトリクス

これで Task #107 deck-builder agent の作業時、 「cards.json + matchups.json」
を読んで card combination → 評価 → deck 出力、 という設計が可能になる。

## deck-builder agent prototype v1 (Task #107、 2026-06-18)

`scripts/build_deck.py` を新規実装、 cards.json を読んで 60 cards を heuristic
で構築。 `--target-type R` で Fire 系を狙うことが可能。

### v1 構築結果 (Fire 系 target)

- **4x Hearthflame Mask Ogerpon ex** (HP 210, type {R})
- **4x Mega Camerupt ex** (HP 340, type {R}, damage 280/1 energy = TOP)
- Buddy-Buddy Poffin x4, Dusk Ball x4
- Carmine x3, Lillie's Determination x3
- Switch x2, Boss's Orders x2
- Hero's Cape x1 (ACE SPEC)
- Gravity Mountain x2 (stadium)
- Basic Fire Energy x31

### 設計上の制限 (= 次の改善ポイント)

1. **進化チェーン整合性が無い**:
   - Mega Camerupt ex は Numel → Camerupt → Mega Camerupt ex で進化
   - 現状は Basic から直接置けると仮定 = 進化先 (Stage1/2) のみ含むデッキ
     は本物の試合では evolve できない
   - 修正: cards.json に `evolves_from` を追加、 evolution chain を構築
2. **evaluate_deck は表面的**: HP 合計、 type 数だけ。 実際の試合性能と
   相関しない
3. **trainer staples が固定**: deck タイプ (attacker 系 / 壁系) によって
   最適な supporter/item は違うはず
4. **Energy 種別が 1 種のみ**: 一部 deck (V6 など) は dual-type で運用
5. **ACE SPEC の有無**: Hero's Cape は wall 系向き、 attacker 系には
   Hyper Aroma など他選択肢が望ましい

### v2 改善方針

1. evolves_from を card DB から抽出 (kaggle_data/EN_Card_Data.csv の
   "Previous stage" 列を読む)
2. evolution chain を deck-builder が辿る → 「Basic + Stage1 + Stage2」 を
   正しく配置
3. 実際に bench で勝率を測る評価関数 (= matchups.json 形式に揃える)
4. GA / RL で deck pool を進化させる loop

## deck-builder agent v2 (2026-06-18、 同サイクル) — 進化チェーン対応

1. `analyze_cards.py --json` の `pokemon_db` に `evolves_from` (= 進化前の
   Pokemon Name) を追加
2. `build_deck.py` の `pick_attacker_line` を書き換え:
   - Stage 1 attacker を damage-eff + HP-eff + meta-fit で score
   - `evolves_from` を name index で解決して、 対応する Basic 印刷を取得
   - lowest card_id の Basic を選ぶ (= base printing)
3. 結果として「進化チェーンが正しい」 deck だけが構築される

### v2 構築例 (Fire target)

  4x Salandit (Basic, HP 70, type {R})
     ↓ evolves_from name="Salandit"
  4x Salazzle ex (Stage 1, HP 260, type {R})

v1 の Mega Camerupt ex (Stage 1, evolves_from Numel) ではなく Salazzle ex
が選ばれたのは、 HP eff (260 / (retreat+1)) の差。 評価関数の重み調整で
Mega Camerupt 路線にも切替可能。

### 次の改善 (v3)

- Stage 2 Pokemon サポート (e.g. Mega Venusaur ex は Bulbasaur → Ivysaur → Mega)
- Trainer staples を deck type に応じて切替 (e.g. Wall 系は Hero's Cape、
  Attacker 系は Hyper Aroma)
- 実 bench で勝率測定 → 進化フィードバックで improve

## deck-builder v3 (Stage 2 chain) 完成 (2026-06-18)

`scripts/build_deck.py:pick_attacker_chain()` を実装:
- candidates に Stage 1 と Stage 2 の両方を含める
- `_resolve_chain()` で 1-2 段の evolves_from を name で遡る
- chain 長に応じてセットアップペナルティ (-5 per evolution step) を score に
  加算 (= Stage 2 が無条件で選ばれないように)

### v3 構築例 (Fire target)

  4x Charmander (Basic, HP 80, type {R})
     ↓ evolves_from name="Charmander"
  4x Charmeleon (Stage 1, HP 110, type {R})
     ↓ evolves_from name="Charmeleon"
  4x Mega Charizard Y ex (Stage 2, HP 360, type {R}) ← 強力 attacker

これで builder は Mega Camerupt ex (Stage 1)、 Salazzle ex (Stage 1)、
Mega Charizard Y ex (Stage 2) など全 stage の attacker を候補にできる。

### 次の v4 改善

- Trainer staples を deck-type-aware に切替 (Wall系は Hero's Cape、
  Attacker系は Hyper Aroma、 setup-heavy は Buddy-Buddy Poffin 重視)
- 実 bench で fitness 測定 (rule-based 相手に 20g 程度)
- GA loop で deck pool を進化させる

## deck-builder v4: 実 bench fitness 評価 (2026-06-18)

`scripts/build_and_eval_deck.py` を実装:
1. 複数 spec (target_type × allow_stage2) で deck を構築
2. **type-neutral heuristic agent** (ATTACH→EVOLVE→PLAY→ABILITY→ATTACK 優先)
   で各 deck を駆動
3. 5 種 rule-based opp 相手に 10g/side で playout
4. overall winrate で ranking

build_deck.py に `--no-stage2` flag も追加。

### 6 spec × 5 opp × 10g (= 300 games) 結果

| rank | spec | overall | Mega Luc | Dragapult |
|---|---|---|---|---|
| **1** | **Fighting / Stage1** | **18.0%** | **40%** | **30%** |
| 2 | Fighting / Stage2 OK | 4.0% | 20% | 0% |
| 3 | Lightning / Stage1 | 2.0% | 0% | 0% |
| 3 | Fire / Stage1 | 2.0% | 10% | 0% |
| 3 | Default / Stage2 OK | 2.0% | 0% | 0% |
| 6 | Fire / Stage2 OK | 0.0% | 0% | 0% |

### 重要な発見

1. **Stage2 は 5g/opp の短期戦で setup 間に合わず壊滅** (Fire/Stage2 = 0%)
2. **Fighting/Stage1 が anti-meta**: Mega Lucario 40%、 Dragapult 30%
3. **type_target='F' は機能していない**: スコアで type bonus が
   eff (HP/retreat) より小さく、 結果として **Water 系の
   Snorunt → Mega Froslass ex (HP 310 Stage1 ex)** が選ばれた
4. **偶然の HP 高 ex Stage1 が anti-meta** という発見:
   - Snorunt id=103 → Mega Froslass ex id=861 (HP 310)
   - Water type は LB の F weakness 323 cards は突けない
   - だが HP 310 のタフネスで Mega Lucario/Dragapult に張り合える
5. これは **deck-builder agent の自己評価ループの最初の発見** —
   人間が思いつかない deck を heuristic が prototype として提案

### 次サイクル方針

- **Mega Froslass ex deck を 80g で本格評価** (lab で 18% は他 subjects と
  比較できる)
- **build_deck.py の score 関数を見直し**: target_type bonus を 30 → 100 に、
  もしくは type bonus を winrate-相関の高い weight に再設計
- **GA loop**: top spec の deck を seed に carbon copy + 1 card 変更で
  fitness が上がるか測る

### Snorunt + Mega Froslass ex 40g/opp 本格評価

  vs Mega Lucario:   12-28 (30.0%)
  vs Dragapult ex:   13-27 (32.5%)
  vs Iono:            3-37 ( 7.5%)
  vs Mega Aboma:      9-31 (22.5%)
  vs Crustle Wall:    7-33 (17.5%)
  vs Crustle Dashi:   0-40 ( 0.0%) ← ex 弱点 (ふしぎなロックイン)
  vs V6:              5-35 (12.5%)
  **overall: 49-231 (17.5%) across 280g**

5g 評価 (18.0%) と 40g 評価 (17.5%) は **ほぼ一致** → fitness signal 自体は
安定している。 ただし 17.5% は subjects 全体で見ると ZoliDragapult (17.1%)
と同水準、 下から 2 位。 3-MLP (23.3%) より **-5.8pp 弱い** ため LB 提出候補
にはならず。

### 重要な含意

1. **「短期 5g 評価で見えた強さ」 は relative ranking として有効** だが、
   absolute winrate は overall 20% を超える deck の発見にはまだ届かない
2. **build_deck.py の score 関数は弱い**:
   - target_type='F' (Fighting) を指定しても無視され Water 系 ex が選ばれた
   - 「HP / (retreat+1)」 が dominant な weight、 type effectiveness は後付
   - 適切な weight 再設計が必要
3. **Crustle Dashi に対する ex 弱点** は build_deck では構造的に解決不可
   (Mega Froslass ex も ex なのでロックインで詰む)
4. **deck-builder v4 の真の価値**: 80g 級でも anti-meta deck を発見する
   ためには:
   - score 関数で **「ex 以外」 のオプションを残す**
   - V6 のような hybrid (ex + non-ex 攻め分け) を表現する

### v5 への方針

- score 関数の重み再設計 (type bonus を 30 → 100、 ex penalty を新設)
- **non-ex Hariyama / Solrock 系の手書き spec** を candidate に加える
- GA loop: best spec を seed に 1 card 変更で fitness 改善を測る

## deck-builder v5 (score 関数改修) — **逆効果** (2026-06-18)

実装:
- `pick_attacker_chain` で target_weakness bonus を 30 → **200** に
- ex penalty **-20** を新設 (Crustle ロックイン対策)
- `require_non_ex` 引数で non-ex Hariyama 系を強制可能に
- `build_and_eval_deck.py` に non-ex 系の 3 spec を追加

### 8 spec × 5 opp × 5g (= 200 games) 結果

| rank | spec | overall | attacker (head) |
|---|---|---|---|
| **v4 比較** | **Fighting/Stage1 (旧)** | **18.0%** | **Snorunt → Mega Froslass ex** |
| 1 | Fighting/Stage1 (v5) | 8.0% | id=333 |
| 2 | Fire/Stage1/non-ex | 6.0% | id=496 (Volcanion?) |
| 3 | Psychic/Stage1 | 4.0% | id=1039 |
| 4-5 | Lightning, Fighting/non-ex | 2.0% | - |
| 6-8 | Fire, Default, Default/non-ex | 0.0% | - |

すべて v4 の 18% から大幅劣化。

### 失敗の原因

1. **target_type bonus 200 が強すぎ**: 「meta-fit の HP 高 ex」 を outranking
   できる Fire type の attacker が無く、 結果として弱い type 強制で
   1025 (Salandit) などが選ばれた
2. **ex penalty -20** は Snorunt+Mega Froslass ex (HP 310 で eff 100+) を
   敗者にし、 Stage1 の弱い non-ex Hariyama (HP 130 程度) が勝てない
3. **強い type への偏重** が「人間が思いつかない」 deck を発見する能力
   (= v4 の核心) を**潰した**

### v6 方針 (本サイクル末尾)

- **v4 score 関数に revert** する (HP/(retreat+1) 主導が正解だった)
- ex penalty は **無効化** or 大幅軽減 (-5 以下)
- target_type bonus は **30** で十分 (200 は強すぎ)
- 真の improvement は **GA loop** で「best spec を seed に 1 card swap で
  fitness 改善を測る」 直接 fitness-driven 進化
- 教訓: heuristic score の手動 tuning は noise vs guidance のバランスが
  難しい、 fitness ベース GA の方が安全

## GA loop 初回実行 — **進化に成功!** (2026-06-18)

`scripts/ga_deck.py` を 10 generations、 3g/eval (= 30g per opp) で実行:

  initial fitness: 10.0% (3-MLP 23.3% より -13.3pp、 noise 大)
  gen 2: Snorunt → Gravity Mountain swap → 16.7% ✓ accept
  gen 3: Water Energy → Metal Energy swap → 23.3% ✓ accept
  gen 4-10: すべて reject (上記 2 つが本質改善だった)

  **final fitness: 23.3% (+13.3pp、 3-MLP と同水準!)**

### 進化後 deck (deck_ga_v1.csv) の構成

  4x Mega Froslass ex (HP 310 Stage1 ex)  ← attacker 維持
  3x Snorunt (Basic、 1 枚減)              ← 4→3 に進化
  Stadium 3 (Gravity Mountain 増)        ← set-up 重視
  Item 10、 Supporter 8、 Tool 1
  Basic Energy (Water + Metal 1 枚)

GA が学んだ含意:
1. **attacker x4 は維持** (gen 8 で Mega Froslass ex 削除を試した → reject)
2. **Basic Pokemon の 4 枚は冗長**、 1 枚 Stadium に置換が利く
3. **Stadium で持続効果** を出す方が単発の Snorunt より valuable
4. Energy 種類変更 (Water 1 枚を Metal に) で何か改善 (noise の可能性も)

### 注意 (3g/eval の noise)

- 3 games/opp/seat × 5 opps × 2 sides = 30 games per fitness 評価
- Wilson 95% CI ≈ ±18pp、 noise 大
- 23.3% は本物の improvement か noise かは **40g+ で再評価必須**
- ただし「方向性」 (= attacker 重要、 Basic 減らせる) は valuable な発見

### 次サイクル方針

1. **deck_ga_v1.csv を 40g/opp で本格 bench** (= noise 除去)
2. **GA を 100 generations + 5g/eval で長期実行** (= scheduler に組込済)
3. 結果が 25%+ なら **submission 候補に**
4. multi-mutation (2-card swap) も試行

### 🚨 GA 40g 本格 bench — improvement は **noise だった**

  vs Mega Lucario:    7-33 (17.5%)
  vs Dragapult ex:    7-33 (17.5%)
  vs Iono:            4-36 (10.0%)
  vs Mega Aboma:      7-33 (17.5%)
  vs Crustle Wall:    5-35 (12.5%)
  vs Crustle Dashi:   0-40 ( 0.0%) ← 持続
  vs V6:              7-33 (17.5%)
  overall:           37-243 (13.2%) ← v4 17.5% から **-4.3pp 悪化**

3g/eval GA evaluation の 23.3% → 40g 本格 bench 13.2% = **改善は noise**。
**Wilson 95% CI ±18pp で覆われる範囲内**。 むしろ「Snorunt → Stadium」
「Water → Metal Energy」 の 2 つの swap は **長期 winrate を下げる方向**
だった。

### GA loop の真の制約 (本サイクル末発見)

1. **3g/eval は使えない**: noise floor が large、 false positive 量産
2. **最低 40g/eval** が必要 (= 1 generation ≈ 14秒 × 5 opps × 8 sides ≈ 60-90秒)
3. **100 gens × 90秒 ≈ 2.5h** → overnight scheduler に適切な workload
4. ただし、 構造的問題 **「単一 ex chain は Crustle 0% 確定」** は GA でも解決不可

### Task #107 deck-builder の構造的限界

- 現状の `pick_attacker_chain()` は **単一 evolution chain** のみ
- ex attacker を選ぶと **Crustle ロックインで詰む** (gen 8 で確認、 v4 80g、 v6 GA)
- 真の anti-meta deck は V6 のような **hybrid (ex + non-ex)**
- builder v7 で hybrid deck をサポートすれば、 Crustle 改善できる見込み

### v7 設計案 (本サイクル末提案)

```python
# 仮: pick_attacker_chain は最大 2 つの chain を返す
def pick_attacker_chains(cards, primary, secondary, ...) -> list[list[dict]]:
    return [primary_chain, secondary_chain]

# Crustle 検出時は secondary (non-ex) を出す agent ロジックも必要
```

これは builder + agent 両方の改修。 中規模実装、 1-2 サイクル必要。

### LB 観察 (2026-06-18)
- V6: **926.5** (横ばい、 我々の best submission 安定)

## GA 8g/eval 再実行 (2026-06-18) — trade-off 発見

3g/eval が noise だった反省で **8g/eval (= 80 games per fitness)** に上げて
15 generations、 rng_seed=7 で実行 (5 分):

  initial: 18.8% (v4 baseline 17.5% とほぼ整合 — noise 圏内に収束)
  gen 2: swap[25] Water Energy → **Boss's Orders (1182)** → 22.5% ✓ accept
  gen 3-15: すべて reject (= 13 連敗、 真の global optimum 付近に居る示唆)

  final fitness (8g eval): 22.5%

### deck_ga_v2 を 40g/opp 本格 bench

  vs Mega Lucario:   6-34 (15.0%) ← v4 30% から **-15pp**
  vs Dragapult:     11-29 (27.5%) ← v4 32.5% から -5pp
  vs Iono:           3-37 ( 7.5%) ← 同水準
  vs Mega Aboma:     8-32 (20.0%) ← -2.5pp
  vs Crustle Wall:   4-36 (10.0%) ← -7.5pp
  vs Crustle Dashi:  0-40 ( 0.0%) ← **持続** (ex 構造的問題)
  vs **V6**:        11-29 (**27.5%**) ← v4 12.5% から **+15pp**
  overall:         43-237 (**15.4%**) ← v4 17.5% から **-2.1pp**

### 重要な発見

1. **Boss's Orders (1182) 追加は deck-specific な効果**:
   - V6 / Dragapult (hybrid 系) **+10-15pp** 改善 (相手の bench を gust で
     引き出して 2 prize KO を狙える)
   - Mega Lucario / Crustle Wall (直接打点系) **-7.5 ~ -15pp** 悪化
     (Boss は Mega Lucario の HP 高に対して 1 turn の手数で勝負を決められない)
2. **8g/eval (80 games)** でも Wilson CI ±11pp、 まだ noise あり
3. GA は「特定 matchup を伸ばすが他を犠牲にする trade-off」 を選びがち
4. **overall を上げるには multi-mutation + Pareto 評価が必要** (= 単一 swap は
   local optimum をすぐ exit する)
5. **Crustle Dashi 0% は完全に構造的** — Mega Froslass ex を使う限り解決不可

### v7-v8 設計案

- **v7**: hybrid chain (primary ex + secondary non-ex) で Crustle 対策
- **v8**: GA loop で multi-fitness (overall + 各 matchup weighted average)
  を最適化、 single matchup の犠牲を penalize
- 真の breakthrough は v7 (hybrid) の方が大きい見込み

## v7 hybrid 実装 + bench — **失敗** (2026-06-18)

`build_deck.py` に `pick_attacker_chains()` (primary ex + secondary
non-ex) と `build_hybrid_deck()` を追加。 `--hybrid` flag で利用可能。

v7 deck (Fire target):
  Primary chain x3: Charmander → Charmeleon → Mega Charizard Y ex (HP 360)
  Secondary chain x2: Croagunk → Toxicroak (HP 130 Fighting non-ex)
  16 Fire Energy + trainer staples

### 30g/opp bench 結果

  vs Mega Lucario:    3.3% ← v4 30% から **-27pp**
  vs Dragapult:       3.3% ← -29pp
  vs Iono:            0.0%
  vs Mega Aboma:      0.0%
  vs Crustle Wall:   23.3% ← v4 17.5% から +6pp
  vs Crustle Dashi:   0.0% ← **持続!** hybrid でも解決せず
  vs V6:              3.3%
  overall:           4.8% ← v4 17.5% から **-12.7pp**

### 根本原因の判明

**「generic_agent は OPTION_PRIORITY 優先順 (ATTACH→EVOLVE→PLAY→ABILITY→
ATTACK→RETREAT) で動くため、 deck に secondary chain が入っていても
primary chain を盲目的に setup する」**

- 結果: Stage 2 Mega Charizard Y ex の setup が間に合わず (Fire/Stage2 既知
  の問題)、 secondary Toxicroak は使われずに deck 内で腐る
- Crustle Dashi 0% も持続 — generic_agent が Crustle を「検出」して secondary
  に routing する logic を持たないため
- **deck-level の hybrid 構造だけでは無意味**、 agent-level の routing が必須

### 次の v8 方針 (本サイクル末発見)

**真の hybrid solution は builder + agent の同時改修**:

1. **builder v8**: `pick_attacker_chains` は v7 のまま (deck 構造 OK)
2. **agent v8**: `make_generic_agent` を **Crustle 検出 + non-ex 切替**
   logic に改修
   - obs から相手 active の card_id を見て Crustle (id=344/345) と
     その派生 ID を検出
   - 検出時は ATTACH/EVOLVE を secondary chain の card に向ける
   - これは V6 の `CRUSTLE_AWARE=True` ロジックと同じ発想
3. **bench で再評価**: agent v8 + deck v7 で Crustle 0% を抜けられるか

これは builder + agent 両方の **真の anti-Crustle solution** への正しい道ですわ。
中規模実装、 次サイクル目標。

## v8 agent-routing 実装 + 真の構造問題発見 (2026-06-18)

### v8 実装内容

`build_and_eval_deck.py:make_generic_agent` に:
- `secondary_card_ids: set[int]` 引数追加
- `_opp_has_crustle(obs)`: obs.current.players[1-yourIndex] の active/bench/
  discard 全てを舐めて Crustle id=344/345 を検出
- Crustle 検出時は `_option_targets_secondary` が True を返す option に
  priority -10 を加算 (= 通常 priority より上位)
- `_option_targets_secondary` は contextCard.id と option.cardId を確認

### v8 vs v7 bench (deck_hybrid_v7、 secondary={679, 680})

  vs Crustle Wall:    v7=6.7%, v8=6.7%  (= 同一)
  vs Crustle Dashi:   v7=3.3%, v8=3.3%  (= 同一)

### 真の構造問題: secondary x2 は引けない

v8 logic 自体は OK だが effect なし、 理由は:
- hybrid deck で secondary chain は **Croagunk x2 + Toxicroak x2 = 4 枚** のみ
- 60 cards 中 4 枚 = 開幕 7 枚 hand に来る確率は 4/60 × 7 ≈ 47%、
  Stage 1 までいくには更に 2-3 turn 必要
- Crustle 戦は 5-10 turn で勝負が決まる、 そこに **Stage 1 + 進化 + ATTACK**
  の 3 turn セットアップが間に合わない
- 一方 V6 は Hariyama (Stage 1) x4 + Makuhita (Basic) x4 を含み、 anti-Crustle
  attacker は **デッキの 13%** を占める

### v9 設計案 (本サイクル末)

1. **build_hybrid_deck の比率変更**:
   - 現状: primary x3 chain + secondary x2 chain = 9-11 + 4 = 13-15 枚
   - v9: primary x2 chain + secondary x3 chain = 6-8 + 6 = 12-14 枚
   - secondary を **4 枚以上** にすれば Crustle 戦の手札確率が改善
2. **もしくは secondary = single Basic non-ex 4 枚** (V6 と同じ構造)
   - Hariyama x4 単独で deck を組む (Basic only、 進化不要なので速い)
3. **v8 routing logic は維持** (deck 比率を上げれば effect 出るはず)

v9 実装 + bench は次サイクル目標。

## v9 — secondary 比率を 4 に増やしても Crustle 0% 持続 (2026-06-18)

`build_hybrid_deck(..., n_secondary_each=4, primary_allow_stage2=False)` で
構築:
- Primary: Salandit x3 → Salazzle ex x3 (Stage 1 Fire ex)
- **Secondary: Croagunk x4 → Toxicroak x4 (Stage 1 Fighting non-ex)**
- 25 Fire Energy + trainer staples

secondary 8 枚 (= deck の 13%) で V6 と同等の比率を確保。

### v9 + Crustle-aware vs no-route @ 30g/opp

  matchup        | no-route | Crustle-aware | delta
  Mega Lucario   |  13.3%  |   23.3%      | +10pp
  Dragapult      |   0.0%  |    0.0%      |   0
  Iono           |   0.0%  |    0.0%      |   0
  Mega Aboma     |   3.3%  |    0.0%      |  -3pp
  Crustle Wall   |   6.7%  |    6.7%      |   0
  Crustle Dashi  |   0.0%  |    0.0%      |   0 (持続)
  V6             |  23.3%  |   20.0%      |  -3pp
  overall        |   6.7%  |    7.1%      | +0.4pp

### **真の構造問題の発見**: V6 成功の本質

secondary 比率を上げても Crustle 0% は **解決せず**。 仮説検証で判明:

V6 が Crustle に強い真の理由は **deck 比率だけではない**:
- 13% Hariyama は **必要条件** だが **十分条件ではない**
- V6 agent の内部 logic に **「Crustle 検出時の active rotation +
  Hariyama 専用 attack 順序選択」** がある
- 具体的には V6 main.py の `CRUSTLE_AWARE=True` の中身は:
  - obs から Crustle 検出時、 active を Mega Lucario ex から Hariyama に rotate
  - Hariyama に Fighting エネルギーを attach
  - Hariyama の attack で Crustle のロックインを抜ける
  - これは **30+ 行の専用ロジック** で、 単純な OPTION_PRIORITY 切替では不可能

我々の generic_agent は **ATTACH→EVOLVE→PLAY 優先順** のみで:
- Crustle 検出は obs 確認できる
- 「Toxicroak を hand から play する」 は detect 可能 (cardId=679)
- ただし **「Salazzle ex から Toxicroak へ active rotation」** が我々の
  agent では特定できない (RETREAT option はあるが、 どの bench に switch
  するかの 2 段階選択を扱えない)

### Task #107 deck-builder agent の構造的限界 (本サイクル末確定)

deck-builder agent (Task #107) で生成した deck を **generic_agent で動かす**
アプローチは、 anti-Crustle の本質を表現できない。 解決には:

1. **agent も deck-builder と同時に生成 / 構築**
   (= 「deck-aware agent generator」)
2. **既存の V6 / Iono / CrustleDashi の専用 agent ロジックを vendor して
   使う** (現在の rule-based submission の路線)
3. **深層学習で agent を訓練** (= 我々の V60 路線、 ただし features 改造
   や value head の問題で 21% の壁)

つまり Task #107 単独では **anti-meta deck の発見はできても、 実戦運用には
agent の改修が並行必要**。 これは大きな pivot point。

### 次サイクル方針 (大幅見直し)

- **deck-builder agent (Task #107)**: 「deck 発見」 と「実用」 を切り分けて、
  deck pool に追加する形に留める
- **V6 superhybrid agent vendor / 解析**: V6 の Crustle 検出 logic を読み解いて、
  generic_agent に組込めるか検討
- **路線の見直し**: rule-based submission 路線 (V6 LB 926.5) を強化する方が
  efficient な可能性

## User 方針確定: (A) + (C) (2026-06-18)

User 選択 = (A) deck-aware agent generator + (C) 深層学習路線。
**(B) rule-based vendor 路線は採用しない** = 我々自身で agent を作り続ける。

## v10 — Crustle 検出時 RETREAT rotation — **逆効果**

`make_generic_agent` に v10 logic 追加:
- Crustle 検出 AND 自分の active が secondary でない時
- RETREAT (option type 12) priority を 4 に boost (= ABILITY と ATTACK の間)
- 続く swap-to-bench select でも secondary を優先

### v10 bench @ 30g/opp

  vs Mega Lucario: 16.7% (v9 23.3% から -6.7pp)
  vs Crustle Wall:  0.0% (v9  6.7% から -6.7pp!)
  vs Crustle Dashi: 0.0% (持続)
  vs V6:           20.0%
  **overall: 5.7% (v9 7.1% から -1.4pp 悪化)**

### 失敗原因の判明: reactive RETREAT は損失

1. **RETREAT は energy cost** を要求 (1-2 energy 消費)
2. RETREAT 中に **opp が attack して prize 取る**
3. secondary も Stage 1 まで進化必要、 場に出てもすぐ attack できない

V6 が成功する真の理由:
- **proactive setup**: 初期 turn から secondary を bench に optimal 配置
- Crustle 検出時には secondary が既に **ready (= Stage 1 になっている)**
- 我々の generic_agent は OPTION_PRIORITY で **「常に primary を優先」**、
  secondary は冷遇される
- これは reactive vs proactive の根本差

### v11 方針 (本サイクル末): proactive deploy

- 初期 turn から secondary も **常時 bench に optimal 配置**
- 「opp が Iono と分かるまで保険として setup」
- 「Crustle と分かったら secondary 全力」
- これは V6 の `CRUSTLE_AWARE=True` の核心ロジック

generic_agent 改修:
- `PLAY` option (= hand から bench に play) で secondary card を **常時最優先**
- 「primary x4 が bench に出るまで secondary は不要」 ではなく
  「secondary x1 が bench に出るまで primary x4 は出さない」

## v11 — proactive secondary deploy — **微改善のみ** (2026-06-18)

`make_generic_agent` に v11 logic 追加:
- `_my_secondary_count(obs)`: 自分の active+bench で secondary chain の
  Pokemon を数える
- secondary 0 体時は **base priority -20** で proactive boost (= ATTACH/
  EVOLVE/PLAY で secondary を最優先)
- secondary 1 体ある後は通常 priority + v8 Crustle boost のみ

### v11 bench @ 30g/opp

  vs Mega Lucario: 23.3% (v9 同水準)
  vs Dragapult:     3.3% (v9 +3pp)
  vs Iono:          0.0% (持続)
  vs Mega Aboma:    6.7% (v9 +7pp)
  vs Crustle Wall:  6.7% (v9 同水準)
  vs Crustle Dashi: 0.0% (**持続**)
  vs V6:           16.7% (v9 -3pp)
  **overall: 8.1% (v9 7.1% から +1.0pp、 noise 内)**

### **手動 logic 路線は頭打ち** — pivot 判断

v8-v11 で deck-aware agent を 4 サイクルかけて手動構築したが、 **Crustle
Dashi 0% は持続**、 overall は v4 baseline 17.5% に届かず。

V6 が成功している真の理由は **30+ 行の専用 anti-Crustle logic** で:
- 初期 select で active を Hariyama 系に
- 中盤で energy を Hariyama に集中
- 「Crustle 検出時の active rotation」 という 2 段階決定を扱う
- これは generic_agent の OPTION_PRIORITY を boost するだけでは表現不可

### User 方針 (A)+(C) の再検討

- **(A) deck-aware agent generator**: 手動 logic 路線は限界が見えた
- **(C) 深層学習路線**: policy が「opp 検出 + secondary route 学習」 を
  自動で embed できる。 V60 EXT は features_v60 で opp の deck-id
  fingerprint も含むので、 学習量を増やせば Crustle 検出 + 適切な action
  選択を学べるかも

### 次サイクル方針: (C) を再優先

- V60 EXT (= ext1, ext2) は 5500-8500ep で振動 → 振動の真因を再調査
- もしくは **「我々の deck.csv + V60 + 8500ep」** で再評価
- もしくは **「V6 deck.csv + V60 + 大量 ep」** で deck-aware policy 学習
- ただし deck.csv 変更は features_v60 の card-id hash bucket に影響あるので
  fresh init 必須

## (C) V60 路線の本格 pursuit (2026-06-18 夕方)

### インフラ整備完了

- `scripts/bench_v60.py`: 任意の V60 .pt を 7 opp で評価 (`--games N` で
  per-side N games = 2N/opp)
- `main_v60.py`: glob で V60 .pt を集めて、 1 つなら single、 2 つ以上なら
  EnsemblePolicyV60 を自動選択
- `train/ensemble_policy_v60.py`: ロジット平均で `.logits()` / `.value()` /
  `.probs()` API を維持
- `make_submission_v60.sh`: bundle build + sandbox verify
- 全て build verified、 EXT3 完了で即提出可能

### 学習量と V60 性能の対応 (20g/opp benchmark)

| version | total ep | overall | best matchup | worst matchup |
|---|---|---|---|---|
| fresh pool5 | 2500 | **17.1%** | Mega Lucario 40% | Crustle Wall 5%, CrustleDashi 5% |
| EXT1 (warm-start) | 5500 | **20.1%** | Mega Aboma 40% | CrustleDashi 5% |
| EXT2 (warm-start round 2) | 8500 | ~20% (振動) | - | - |
| EXT3 (lr=1e-4 round 3) | 10500 | **学習中** | TBD | TBD |

**学習量で +3pp 改善** (fresh→EXT1) を確認、 学習継続には意味あり。
ただし EXT3 で 25%+ は **microscopic gain** 期待 (~23-24% 程度予想)。
振動 (ext2 で 8500ep でも 20% 程度) も警戒。

### V60 EXT3 完了後のフロー

```bash
# 1. solo bench
scripts/run.sh python3 scripts/bench_v60.py --weights train/mlp_policy_v60_ext3.pt --games 20

# 2. もし 25%+ なら ensemble bench も (= EXT1 + EXT3 の自動 2-V60 ensemble)
scripts/run.sh python3 scripts/bench_v60.py --weights train/mlp_policy_v60_pool5_ext.pt --games 20
# (上記は main_v60 が自動的に ensemble 化するので bench としては single でも OK)

# 3. 提出
./make_submission_v60.sh
.venv/bin/kaggle competitions submit -c pokemon-tcg-ai-battle \
    -f submission_v60.tar.gz -m "V60 EXT3 (10500ep) — first deep-learning submission"
```

### EXT3 完了 + 初の deep-learning submission (2026-06-18 夕方)

**学習結果**: 5000ep × ~340ms/ep = 28 分、 recent 0.17-0.25 で振動。

### single + ensemble bench @ 30g/opp (210 games)

| matchup | EXT1 (5500) | EXT3 (10500) | EXT1+EXT3 ens |
|---|---|---|---|
| Mega Lucario | 35% | 13% | 13% |
| Dragapult | 25% | **37%** | 23% |
| Iono | 10% | 10% | 10% |
| Mega Aboma | **40%** | 37% | 20% |
| Crustle Wall | 15% | **30%** | 27% |
| Crustle Dashi | 5% | 3% | **10%** |
| V6 | 10% | 13% | **23%** |
| **overall** | **20.1%** | **20.5%** | **18.1%** |

### 重要観察

1. **EXT3 と EXT1 は overall ほぼ同水準** (+0.4pp) だが **matchup mix が違う**:
   - EXT3 は Dragapult/Crustle Wall +、 Mega Lucario -
   - 学習継続で「特定 matchup に偏った policy」 になった (振動の結果)
2. **ensemble は overall -2pp 悪化**: matchup mix の中庸化が逆効果
   - ただし Crustle Dashi +5pp、 V6 +10pp の **個別改善** あり
   - EXT1 の Mega Aboma 40% を ens が引き下げて 20% に
3. **3-MLP 23.3% (80g) に届かない**: noise floor を考慮すれば実質同水準だが、
   LB 評価で **3-MLP (679.6) と同程度の 700 前後** が予想

### 53810836: V60 EXT3 single を submit (user 判断)

User 選択 = 「反さず submit して記録に残す」。 single 20.5% (ensemble 18.1%
より強い) を bundle、 EXT1 は train/archive/ に退避。

```bash
.venv/bin/kaggle competitions submit -c pokemon-tcg-ai-battle \
    -f submission_v60.tar.gz \
    -m "V60 EXT3 single (10500ep, lab 20.5%) — first deep-learning LB submission"
```

ref **53810836** PENDING、 評価結果は次サイクル以降。 これは我々自身の
**深層学習 agent の初の LB 投稿** という記録に意味あり (V6 921 を超える
見込みは皆無、 3-MLP 並みの 700 前後を期待)。

### 🚨 53810836 → ERROR (2026-06-18 夜)

V60 EXT3 submission が **Kaggle で動作失敗**。 publicScore 0、 ERROR。

ローカル sandbox check (= scripts/check_main_exec.py 相当)で:
- bundle 構造 OK (deck.csv、 train/{features_v60, mlp_policy_v60, ext3.pt})
- _POLICY load 成功 (MlpPolicyV60 with pi=(64,32) v=(32,))
- agent(deck-submission obs) → 60 cards
- agent(select obs) → [1]

→ **ローカルでは完全動作、 Kaggle 環境固有の crash**

### 仮説 (次サイクルで検証)

1. **Kaggle ランタイムに torch がない** → `import torch` 失敗
   - main_v60 の `_try_load_v60` は except 済みなので、 _POLICY=None で
     engine-prior fallback になるはず
   - ただし、 fallback path に何か bug があるかも (= 確認必要)
2. **試合中の torch 推論が timeout**
   - Kaggle CPU で MlpPolicyV60.logits() が遅い、 1 ターン制限 (~3秒) 超過
3. **main_v60 の deck.csv 読み込み path 問題**
   - `_HERE = Path.cwd()` (= __file__ なしの場合) が想定外の場所
4. **bundle の何かが Kaggle と非互換**
   - tar 構造、 encoding 等

### 次サイクル方針

1. **kaggle.com で 53810836 のエラーログを直接確認**
2. ローカルで「torch 環境を消した sandbox 試験」 を実装
3. 必要なら **main_v60 を完全 PyTorch-free 化** (numpy のみで V60 推論)
   - mlp_policy_v60.py の forward を pure numpy 版にすれば torch 不要

### ✅ Task #111 解決: numpy-only V60 で再 submit (2026-06-18 夜)

torch 依存を完全排除した V60 を実装、 53810836 ERROR の根本原因と推定して
回避策を実装:

1. **`train/mlp_policy_v60_numpy.py`** 新規: pure-numpy 実装の `MlpPolicyV60Numpy`
   - `_pi_forward`: matmul + ReLU 手書き
   - `_v_forward`: 同様、 tanh も numpy.tanh
   - `state_dict` 形式 ({pi.0.weight, pi.0.bias, ...}) を numpy 配列に変換
2. **`scripts/extract_v60_weights.py`**: torch .pt → numpy .npz
   - build time に走り、 torch dependency を build 環境に限定
3. **`main_v60.py`** 改修: glob `mlp_policy_v60*.npz` を見て numpy 版を load
4. **`make_submission_v60.sh`** 改修: .pt → .npz 抽出を bundle 化に統合、
   ship する train/ には `mlp_policy_v60_numpy.py` + `.npz` のみ
   (torch 系ファイルは除外)

### Parity check

  numpy logits: [-4.9971819 -2.6859689]
  torch logits: [-4.9971814 -2.6859689]
  max abs diff: 4.77e-07 (float32 精度内、 **実質一致**)

### sandbox test

  bundle contents: cg, deck.csv, main.py, train
  train/: __init__.py, features.py, features_v60.py,
          mlp_policy_v60_ext3.npz, mlp_policy_v60_numpy.py
  _POLICY type: MlpPolicyV60Numpy
  agent(select obs): [1] ← argmax 動作確認

### Re-submit: 53812115

```
.venv/bin/kaggle competitions submit -c pokemon-tcg-ai-battle \
    -f submission_v60.tar.gz \
    -m "V60 EXT3 numpy-only (retry of 53810836 ERROR; pure-numpy inference, no torch needed)"
```

ref **53812115** PENDING、 評価結果は次サイクル以降。 これが COMPLETE して
スコアが出れば「我々の deep-learning agent が Kaggle で動く」 ことを実証、
torch 依存削除戦略の正しさを確認。

### 🚨 53812115 も ERROR! (= torch 依存仮説は **不正解**)

numpy-only 化しても Kaggle で ERROR。 真因は別。

### 真因発見: deck.csv path resolution

3-MLP main.py (LB 679.6 で動く) と main_v60 を diff した結果、 **deck.csv の
path 解決ロジックが違う**:

3-MLP main.py:
```python
path = "deck.csv"  # try relative first
if not os.path.exists(path):
    path = "/kaggle_simulations/agent/deck.csv"
```

main_v60 (両 ERROR 版):
```python
_HERE = Path(__file__).resolve().parent  # or Path.cwd() if __file__ missing
deck_path = _HERE / "deck.csv"  # one-shot only
```

Kaggle ランタイムが exec() で main.py を呼ぶ時、 **`__file__` 無し** で
**cwd が想定外** の組合せがあり得る。 main_v60 は **single-shot path 解決**
だったので一度ミスると ERROR。 3-MLP は **3 候補 fallback** で頑健。

### Fix #2: 53812882 (multi-root fallback)

`main_v60.py:_candidate_roots()` を実装:
- `Path(__file__).parent` (= sandbox 想定)
- `Path.cwd()` (= __file__ 欠落時)
- `Path("/kaggle_simulations/agent")` (= Kaggle 標準 mount path)

3 ヶ所すべて試して deck.csv + train/ の存在確認。 sys.path にも全部追加。
ローカル sandbox で:
- `_POLICY: MlpPolicyV60Numpy`
- `deck head: [721, 721, 722]`

Re-submit: ref **53812882** PENDING、 3 回目の正直なるか。

### 教訓

main_v60 を実装した際、 3-MLP main.py の **path fallback パターンを継承
しなかった** ことが二重 ERROR の根本原因。 deep-learning 移植時に「ロード
の頑健性」 を軽視した。 今後 submission を増やす時は 3-MLP の
`_read_deck()` をそのまま使い回す方針。

## 🎉 53812882 COMPLETE! V60 deep-learning agent が動いた (2026-06-19)

```
ref     | status    | publicScore | 修正点
53810836 | ERROR    | -          | torch + single-shot path
53812115 | ERROR    | -          | numpy-only だが single-shot path
53812882 | COMPLETE | **600.0**  | numpy + multi-root fallback ✅
```

publicScore **600.0** は TrueSkill 初期値 μ₀、 評価試合 1-2 件のみ完了。
前回 Iono の 600 → 615 → 762 と同じパターン、 試合数が増えれば真の
スコアに収束する。

### 達成: 我々自身の deep-learning agent の初の真の LB 到達

User 方針 (C) の最初のマイルストーン:
- **deck.csv = 我々のオリジナル Mega Abomasnow ex**
- **policy = V60 EXT3 (= features_v60 + 5500ep warm-start from EXT1)**
- **lab 20.5% @ 30g/opp** (3-MLP 23.3% にやや劣るが同水準)

期待値: LB 3-MLP 並みの 700 前後に収束。 V6 926 には届かないが、
deep-learning route が動くこと自体が大きな意味。

### 📉 53812882 評価進行: 600.0 → 523.1 (期待外れ)

V60 EXT3 が動いてから 30 分後、 publicScore は **600.0 → 523.1 に下落**:
- TrueSkill 初期 μ₀=600 から **-76.9 ポイント** = 評価試合で負け越し
- 3-MLP 679.6 と比較して **-156 ポイント弱い**
- lab の **2.8pp 差 (23.3% vs 20.5%) が LB で 156 ポイント差** に増幅

### 重要な学び: lab → LB 変換は非線形

| agent | lab @ 80g | LB |
|---|---|---|
| Iono | 64.0% | 762.2 |
| CrustleDashi | 67.3% | 870-890 帯 |
| V6 | 57.9% | 896-926 帯 |
| **3-MLP** | **23.3%** | **679.6** |
| **V60 EXT3** | **20.5%** | **523.1** |

- 3-MLP → V60 EXT3 は lab -2.8pp で LB -156
- これは **lab 23.3% 以下の policy は LB 競争力ゼロ** を示唆
- V60 路線で 3-MLP を超えるには **lab 25%+ 必須**

### V60 路線の判断点

V60 EXT3 単独は **3-MLP に劣る**。 次の打ち手:

1. **V60 ensemble (= 多 seed)** — 短期、 +3pp 期待:
   - 3-MLP は seed 0/2/100 のアンサンブルで 23.3%
   - V60 EXT3 + 別 seed V60 で 22-24% 目指す
   - Task #112 で seed=31415 を fresh 4000ep 学習中
2. **PPO** — 中長期、 振動制御で 25%+ 期待
3. **deck 入替** — 我々の Mega Aboma deck は弱い、 V6 deck で学習し直す
   (ただし前回 V6 deck fresh 4000ep は 13.9% で失敗)

時間効率重視で (1) を優先試行。 完了後 ensemble bench、 22%+ なら再 submit。

### Task #112 結果: V60 ensemble は機能しない (2026-06-19 朝)

seed=31415 fresh 4000ep 完了 (22 分、 累計 16.8%)。

**single bench**:
- seed=31415: overall 16.2% (Mega Lucario 30%、 Dragapult 27% は強いが Aboma 10%)
- EXT3: 20.5% (前回測定)

**ensemble (EXT3 + seed31415) bench**:
  vs Mega Lucario:  7-23 (23.3%) ← EXT3 +10pp
  vs Dragapult:    10-20 (33.3%)
  vs Iono:          0-30 ( 0.0%) ← **致命的中庸化**
  vs Mega Aboma:    7-23 (23.3%) ← EXT3 -14pp
  vs Crustle Wall:  8-22 (26.7%)
  vs Crustle Dashi: 1-29 ( 3.3%)
  vs V6:            6-24 (20.0%) ← EXT3 +7pp
  **overall: 39-171 (18.6%)** ← EXT3 single 20.5% から **-1.9pp 悪化**

### V60 ensemble の構造問題

前回 EXT1+EXT3 ensemble (-2pp) と今回 EXT3+seed31415 ensemble (-1.9pp) で
**同じ pattern**:
- 個別 matchup で改善 (V6 +7pp、 Mega Lucario +10pp) と悪化 (Iono -10pp、
  Aboma -14pp) が trade-off
- 平均すると -2pp の中庸化

仮説:
- **features_v60 共通**で訓練された policy は本質的に同じ「meta」 を学ぶ
- seed の違いでは matchup-specific な偏りが補完されるが、 overall
  の方向では同調する
- 3-MLP は v40 features で seed 0/2/100 ensemble なら 23.3% に届くが、
  V60 では同じ trick が効かない
- features の表現力が seed-level diversity を圧倒している可能性

### Task #112 close、 next pivot

V60 ensemble 路線は dead-end と確定。 次の打ち手候補:

1. **3-MLP の features を v60 にアップグレード**:
   - 既存 mlp_policy / seed2 / seed100 を features_v60 で fresh 学習
   - 3-MLP の seed-level diversity を v60 features で再現する試み
2. **PPO**: 振動制御で 25%+ を目指す中長期路線
3. **V60 EXT3 を deck.csv = V6 deck で fresh 大量学習**: 前回 4000ep は
   失敗したが 10000ep+ なら収束する仮説
4. **V60 EXT3 + 3-MLP のクロス ensemble**: features 不一致で結合不可、
   別 main.py が必要

### scripts/check_main_exec.py に --strict-cwd 追加

今後の submission ERROR を **submit 前に local で検出** する体制:
- `--strict-cwd`: cwd を sandbox の親に設定 + sys.path に sandbox 入れない
- 53812882 で動くようになった main_v60 v3 でも strict-cwd では ERROR
  (= 過度に厳しい、 Kaggle 実環境とは違うらしい)
- Kaggle ランタイムは **bundle dir を cwd にしている** (= 3-MLP の relative
  path が動く事実から推定)
- ただし「path fallback あり」 が安全な実装

## GA loop 8g/eval × 20 gens 結果 — **構造的限界の再確認** (2026-06-18 夜)

`data/sweep/ga_40g_v1.json` に persist。 376 秒 (6 分) で完了:

  initial: 11.2% (5g/eval noise の引き)
  gen 2: Mega Froslass ex (861) → Premium Power Pro (1141) → 16.2% ✓
  gen 5: Water Energy (3) → Lightning Energy (4) → **20.0%** ✓
  gen 7: Gravity Mountain → Premium Power Pro (tie at 20.0%) ✓
  gen 8-20: すべて reject (13 連敗、 local optimum)
  best fitness (8g/eval): 20.0%

### 40g 本格 bench の真の値: **13.2%**

deck_ga_v3.csv の 40g bench:
  vs Mega Lucario:  6-34 (15.0%)
  vs Dragapult:    11-29 (27.5%)
  vs Iono:          1-39 (2.5%) ← 致命的悪化
  vs Mega Aboma:    8-32 (20.0%)
  vs Crustle Wall:  6-34 (15.0%)
  vs Crustle Dashi: 0-40 (0.0%) ← 持続
  vs V6:            5-35 (12.5%)
  overall:        37-243 (**13.2%**) ← v4 baseline 17.5% から -4.3pp

### 過去 3 回 GA の真の天井

| GA version | eval 内 | 40g 本格 |
|---|---|---|
| v1 (3g/eval) | 23.3% | 13.2% |
| v2 (8g/eval) | 22.5% | 15.4% |
| v3 (8g/eval) | 20.0% | **13.2%** |

→ **GA は 13-15% の天井に張り付く**、 v4 baseline 17.5% を超えられず。

### 構造的限界の判明

1. **1-card swap は local optimum で詰む**: gen 5 で water → lightning
   energy 1 枚交換した時点で improvement が止まる、 同じ deck の異なる
   trainer 配分しか試せない
2. **真の improvement 5pp 以下は 8g/eval (CI ±11pp) で検出不能**
3. **GA は v4 builder の初期構築を改善しない** = builder のコア設計
   (HP/(retreat+1) で attacker 選定) が **既に local optimum**
4. Crustle Dashi 0% は構造的問題で deck-builder では解決不可
   (ex pokemon を入れる限り)

## EXT4 — lr=5e-5 は逆効果 (2026-06-18 夜)

`mlp_train_v60 --lr 5e-5 --warm-start ext3` で 5000ep、 合計 15500ep。
recent 振動 0.21-0.26、 lr=1e-4 (EXT3) と **同じ振動範囲**。

### EXT4 bench @ 30g/opp

| matchup | EXT3 (lr=1e-4) | **EXT4 (lr=5e-5)** | delta |
|---|---|---|---|
| Mega Lucario | 13.3% | 10.0% | -3pp |
| Dragapult | 36.7% | 23.3% | -14pp |
| Iono | 10.0% | 10.0% | 0 |
| Mega Aboma | 36.7% | 23.3% | -14pp |
| Crustle Wall | 30.0% | 20.0% | -10pp |
| Crustle Dashi | 3.3% | 0.0% | -3pp |
| V6 | 13.3% | 6.7% | -7pp |
| **overall** | **20.5%** | **13.3%** | **-7.2pp** |

### 失敗の原因

1. **lr 下げても振動は止まらず** (recent 0.21-0.26 で EXT3 と同じ範囲)
2. **学習過程の peak/valley を制御できない** — checkpoint は last-ep を
   保存するので、 偶然「谷」 で終わると劣化を保存する
3. EXT4 archive 行き。 EXT3 が引き続き V60 best として扱う

### 振動の真因 (推定)

- REINFORCE の policy gradient variance が大きい (= reward sparsity)
- 1 試合で reward は ±1 だけ、 trajectory 内の credit assignment が困難
- lr を下げても variance 自体は減らない
- 解決策:
  - **PPO** (clipped surrogate objective + value baseline GAE)
  - **A2C/A3C** (advantage 推定で variance 削減)
  - **early stopping** + best-checkpoint 保存 (heuristic)
- これらは large refactor、 next phase

## V60 の状況 (本サイクル末)

- EXT3 = **20.5% @ 30g** が我々の deep-learning agent の best
- EXT3 を Kaggle submit → **ERROR (53810836)** 原因究明 task #111 で進行中
- EXT4 で改善試みたが -7.2pp、 archive 行き
- 振動制御には PPO 級の改修が必要、 直近では実装困難

### 次サイクル方針

1. **Task #111 — EXT3 ERROR 原因究明** (最優先): main_v60 を numpy-only 化
   して V60 を実機で動かす
2. ERROR 解決後、 EXT3 を再 submit
3. PPO 路線は中長期目標 (refactor 大)
4. deck-builder GA は構造的限界判明、 投資中止

### Task #107 deck-builder の最終総括

3 段階の GA + v7-v11 hybrid 改修 + v8-11 agent-routing 改修を経て:
- **v4 baseline (= score 関数 + 1 chain) が GA の到達点 17.5%**
- **deck-only の自動 search では LB 競争力 (V6 921) には届かない**
- 真の breakthrough は **deck + agent + 学習** の三位一体改修が必要

### LB 観察 (2026-06-18 夕方)
- V6: 896.5 (前 921 から微減)
- CrustleDashi: **888.2** (前 866 から **+22 上昇!**)

## EXT3 評価中の並行作業 (2026-06-18 夕方)

EXT3 submission (53810836) は PENDING、 LB 評価に数十分必要。 待ち時間に
2 つの background job を並行起動:

### Job 1: EXT4 学習 (GPU、 28 分予測)

`mlp_train_v60` で EXT3 から warm-start、 **lr=5e-5** (= EXT3 の lr=1e-4 の
半分) で振動を更に抑制を試す。 5000ep、 seed=2718。 合計 15500ep を目指す。

仮説: ext1 (5500ep, lr=5e-4) → ext3 (10500ep, lr=1e-4) で recent 振動 0.17-
0.25 だった。 lr を更に下げて 0.20-0.24 に収束を狙う。 15500ep 完了で 22%
台に進めるかも (~25% は微妙)。

### Job 2: GA loop 20 gens @ 40g/eval (CPU、 30-45 分予測)

`ga_deck.py` で Snorunt+Mega Froslass ex deck から 1-card swap 進化。
**8g/eval (= 40g per opp)** で前回の 3g/eval の noise (false positive 量産)
を排除。 20 generations は前回 8g/eval × 15 gens で gen 2 で local optimum
に達した経験を踏まえ、 もう少し脱出を試す。

CPU で走るので EXT4 学習 (GPU) と独立。 結果は `data/sweep/ga_40g_v1.json`
に persist。 generation 毎に dump するので途中で killed しても安全。

### 整合性チェック

EXT1 を `train/archive/` に退避した影響を確認:
- 3-MLP submission (= train/mlp_policy*.pt の v40 系) は **verify OK 維持**
- V60 submission (= train/mlp_policy_v60_ext3.pt のみ) も **verify OK**
- main.py の glob `train/mlp_policy*.pt` は v60 ファイルも match するが、
  EnsemblePolicy が features 不一致でスキップして 3-MLP として動く (= 既知挙動)

## V60 (features_v60.py) 初版学習結果 (2026-06-18)

`train/features_v60.py` (STATE_DIM=60、 deck-ID fingerprint 16+4 buckets) と
`train/mlp_policy_v60.py` / `train/mlp_train_v60.py` を新規実装。 v40 と並存可能。

### 学習: pool5 fresh 2500ep, lr=5e-4, seed=42

- 5 種 opp [Iono, CrustleDashi, V6, Lucario, Dragapult] でローテーション
- recent 勝率: ep 250: 0.11 → ep 2500: 0.19 (上昇傾向)

### V60 solo @ 30g

| matchup | result |
|---|---|
| Mega Lucario | 23.3% |
| Dragapult ex | 20.0% |
| Iono | **3.3%** ← 改善せず |
| Mega Abomasnow | 20.0% |
| Crustle Dashi | **6.7%** ← 改善せず |
| V6 | 13.3% |
| **overall** | **14.4%** |

V40 pool5 (warm-start 1500ep) ≈ 16.7% と同水準、 ノイズ内で差なし。

**features60 の「相手識別」効果は確認できなかった**。仮説:
1. fresh 2500ep では features60 容量に対して **学習不足** (3000-5000ep 必要)
2. bucket hash だけでは「相手の active が Wattrel = Iono」を識別するシグナル弱
3. policy 容量 (64-32) が features60 に対し不足

### 次サイクル方針

User の「単一 agent 強化」方針に従い:

1. **v60 warm-start で更に 2500ep 学習** (合計 5000ep) → 収束させる
2. policy 容量 128-64 に拡張 + warm-start
3. value baseline を 0/1 + EMA に改修 (hard matchup gradient 安定化)
4. PIMC + 学習 policy value head (本命、 並行)

### V60 EXT (warm-start 3000ep、 合計 5500ep) 結果 ✨

  vs Mega Lucario:    8-22  (26.7%)  ← +3.4pp vs 2500ep
  vs Dragapult ex:    8-22  (26.7%)  ← +6.7pp
  vs Iono:            4-26  (13.3%)  ← **+10pp**
  vs Mega Abomasnow:  9-21  (30.0%)  ← +10pp
  vs Crustle Dashi:   4-26  (13.3%)  ← +6.6pp
  vs V6:              5-25  (16.7%)  ← +3.4pp
  overall:           38-142 (21.1%)  ← **+6.7pp**

**全 matchup で改善、 features60 効果を確認できた**。
3-MLP @ 80g 23.3% との差は **-2.2pp** まで縮まった (= 単独 v60 policy で
ensemble なしの 3 つ分に肉薄)。

特に大きな改善:
- Iono: **3.3% → 13.3% (+10pp)** ← 改善のサイン、ただし依然厳しい
- Mega Aboma: 20% → 30% (+10pp)
- Crustle Dashi: 6.7% → 13.3% (+6.6pp)

これで「features60 単独で意味ある signal は無い」という前回判定を**訂正**:
**学習量を増やせば features60 効果は出る**。 fresh 2500ep が短すぎた。

### 次の打ち手 (本格)

1. **更に warm-start 継続** (合計 8000-10000ep) → 3-MLP 23.3% を超えるか
2. **複数 seed で V60 学習 → V60 ensemble 作成** (3-MLP の ensemble 効果を v60 で再現)
3. **policy 容量増 (128-64)** で representation 強化
4. **submission 化**: v60 policy を `main_learned_v60.py` でラップして submit
   候補に加える (User 指示「単一 agent 強化」の到達目標)

### value baseline 修正 (linear-value flag、 2026-06-18)

EXT2 振動の主因と推定する `tanh(V(s))` を bypass できるオプションを追加。
`train/mlp_train_v60.py --linear-value`:
- 従来: `v_pred = torch.tanh(policy.v(sf))` → V(s) は [-1, 1] に拘束
- 改修: `v_pred = policy.v(sf)` → V(s) は任意の実数
- reward は ±1 のままで、 advantage = reward - V(s) が自由に動ける

仮説: hard matchup (5-15% 勝率) では V(s) は **絶対値が 1 を超える負値**を取りたい
(= 「ここから挽回不能」を表現)が、 tanh で -1 に張り付いて表現力を失っていた。
linear にすれば advantage が gradient signal を正しく伝える。

fresh init で seed=7、 3000ep、 lr=5e-4 で 学習中 (background)。

### linear-value 結果 — **失敗**

  vs Mega Lucario: 1-29 ( 3.3%) ← **-23pp** vs tanh EXT1
  vs Dragapult:    7-23 (23.3%)
  vs Iono:         1-29 ( 3.3%) ← -10pp
  vs Mega Aboma:   6-24 (20.0%) ← -10pp
  vs Crustle Dashi: 0-30 ( 0.0%) ← -13pp
  vs V6:           2-28 ( 6.7%) ← -10pp
  overall:        17-163 ( 9.4%) ← **-11.7pp from tanh EXT1**

仮説の **訂正**:
- 当初仮説: 「tanh は hard matchup で V(s) を ±1 にクリップ → advantage 壊す」
- 実証結果: linear V(s) は **学習が遅く、 結果も悪い**
- 実は tanh は **必要な regularization** だった
  - V(s) が無拘束だと早期に exploding values で value loss が暴走
  - value MSE が policy loss を喰い、 policy update が小さくなる
  - 3000ep fresh では収束しない

linear-value policy を `train/archive/mlp_policy_v60_linval.pt` に退避。
ext1 (5500ep tanh) が引き続き作業ベースライン。

### 真の振動原因の再考

EXT2 振動の原因は tanh ではなく以下のいずれかと推定:
1. **lr 設定の不適切** (1e-4 でも reward signal が間欠的に大きい)
2. **policy 容量 (64-32) 不足** — features60 が要求する表現力に届かず、
   局所最適間を「行き来」している
3. **opponent pool 5 が多すぎ** — 各 opp に対する gradient signal が薄まる、
   policy が局所最適を選ぶ事ができない

次サイクル候補 (優先順):
1. **policy 容量 128-64 へ拡張** (= warm-start 不可、 fresh init 必要、 時間コスト大)
2. **lr scheduling** (warm-up + decay)
3. **PPO** (clipped surrogate objective、 variance 削減)
4. **PIMC** (本命、 別軸)

### big (128-64 policy + 64-32 value) fresh 3000ep 結果

  vs Mega Lucario:    4-26 (13.3%) ← small EXT 26.7% より -13.4pp
  vs Dragapult ex:    7-23 (23.3%) ← -3.4pp
  vs Iono:            4-26 (13.3%) ← 0
  vs Mega Aboma:      9-21 (30.0%) ← 0
  vs Crustle Dashi:   0-30 ( 0.0%) ← **6 連続 0%!**
  vs V6:              6-24 (20.0%) ← +3.3pp
  overall:           30-150 (16.7%) ← -4.4pp from small EXT 21.1%

big も fresh 3000ep では収束不足、 small EXT 5500ep に届かず。
容量を増やしても学習量と比例して必要なので、 fresh 3000ep では十分な収束ない。

### 🚨 Crustle Dashi 0% の構造的限界

これまで 6 連続 v60 policy 学習で、 Crustle Dashi vs:

| policy | win rate |
|---|---|
| small fresh 2500ep | 6.7% |
| small EXT 5500ep | 13.3% |
| small EXT2 8500ep | 0% |
| LINVAL fresh 3000ep | 0% |
| **BIG fresh 3000ep** | **0%** |

Crustle Dashi は「ふしぎなロックイン」 (ex Pokemon の attack を 0 にする) +
Jumbo Ice Cream (回復 80) + Hero's Cape で **詰み構造** を作る。

我々の deck.csv は Mega Abomasnow ex 軸 (= ex pokemon) なので、 active ATK が
完全に通らず、 V60 policy が「ABILITY や DISCARD で抜ける」 戦略を学べていない。

これは **deck × policy combination の根本問題**。 解決策候補:
1. **deck を non-ex 化** (我々のデッキ自体を Hariyama 等の non-ex に変える、
   ただし overall 性能が落ちる可能性大)
2. **option type 16 SPECIAL_CONDITION / 11 DISCARD を policy が選ぶよう features 強化**
3. **deck-builder agent (Task #107) で anti-Crustle 構築**
   (Fire 系 attacker、 V6 のような Hariyama 採用パターン)

「単一 policy 強化」 路線では Crustle Dashi 問題は features と deck 両方の
変更なしでは突破できない見込み。 User 方針再確認が必要。

### 🎯 deck 切替 cross-bench (2026-06-18) — 大発見

V60 EXT policy (= 我々 deck で訓練) を **異なる deck で動かして** bench:

| deck | Mega Lucario | Crustle Dashi | Iono | subtotal (60g) |
|---|---|---|---|---|
| 我々 (Mega Aboma) | 30.0% | 5.0% | 0.0% | 11.7% |
| **V6** (Lucario+Hariyama) | **50.0%** | 0.0% | 5.0% | **18.3%** |
| Mega Lucario (Kiyota) | 30.0% | 0.0% | 10.0% | 13.3% |

含意:
- **policy は同じでも deck によって試合性能が大きく変わる**
- V60 + V6 deck で Mega Lucario **+20pp** (30% → 50%)、 subtotal +6.6pp
- Crustle 0% は deck 切替えでも持続 → policy が Hariyama 戦略を学んでいない
  (V6 rule-based は hardcoded で持つ logic)
- User の「deck 進化」 方針が **完全に支持される** 結果

### 進行中: V60 + V6 deck 学習

`train/mlp_train_v60.py:_read_deck()` に `POKE_AI_TRAIN_DECK` env var を
追加。 deck.csv (= 既存 submission に影響) を編集せずに、 任意の deck で
policy 学習可能。

実行中: V6 deck (= deck_romanrozen_v6.csv) で fresh 4000ep、 pool5、 lr=5e-4。

期待: V60 + V6 deck で訓練すれば、 policy は Hariyama 系の戦略を学習し、
Mega Lucario 50% + Crustle 改善 + overall 30%+ になる可能性。
これが当たれば、 main.py + deck_romanrozen_v6.csv で **新提出候補** に。

### V60 V6DECK (fresh 4000ep, lr=5e-4) 結果 — **失敗**

  vs Mega Lucario:    8-22 (26.7%) ← cross-bench で 50% 期待 → 大幅低下
  vs Dragapult:       2-28 ( 6.7%) ← 致命的悪化
  vs Iono:            3-27 (10.0%) ← 改善せず
  vs Mega Aboma:      1-29 ( 3.3%) ← 致命的
  vs Crustle Dashi:   1-29 ( 3.3%) ← 致命的
  vs V6 (mirror):    10-20 (33.3%)
  overall:           25-155 (13.9%) ← V60 EXT 21.1% より -7.2pp

仮説の **訂正**:
- 当初仮説: 「V6 deck で訓練すれば Hariyama 戦略を学ぶ」
- 実証結果: fresh 4000ep では収束不足、 各 matchup で大幅劣化
- 「V6 deck = Lucario+Hariyama hybrid」 は **戦略空間が複雑**、
  4000ep self-play では policy が active rotation や non-ex route などの
  細かい判断を学べない

**深い洞察 (= 新たな方針候補)**:
- 前サイクルの cross-deck bench (V60 EXT + V6 deck = 50% Mega Lucario) は、
  「我々 deck で訓練した policy が V6 deck **でも** ある程度機能する」
  証拠だった
- つまり「提出 deck = V6、 policy = 我々 deck で訓練 」 という分離が
  実は有望
- 実装: main.py が deck.csv (= V6 deck の rename copy) を提出し、
  policy は train/mlp_policy.pt (= 我々 deck で訓練済み) を使う
- これだと既存 3-MLP submission を deck 入替えだけで強化できる

V6DECK policy を `train/archive/` に退避。
EXT (5500ep tanh、 我々 deck) が引き続き作業ベースライン。

### 3-MLP + V6 deck 検証 (2026-06-18)

「policy は our deck で訓練、 提出 deck だけ V6 に切替」 案を 3-MLP で
試行 (cross-bench 20g/opp、 ensemble.members=3):

| matchup | 3-MLP + 我々 deck | + V6 deck | delta |
|---|---|---|---|
| Mega Lucario | 25.0% | **50.0%** | **+25pp** |
| Dragapult | 25.0% | 5.0% | -20pp |
| Iono | 15.0% | 10.0% | -5pp |
| Mega Aboma | 25.0% | 10.0% | -15pp |
| Crustle Dashi | 10.0% | 0.0% | -10pp |
| V6 | 20.0% | 25.0% | +5pp |
| **overall** | **20.0%** | **16.7%** | **-3.3pp** |

- Mega Lucario **+25pp** は deck.csv 入替えの純粋な効果 (V6 deck は
  Mega Lucario と同じ Fighting エネルギーで mirror に近い、 ex 弱点を
  突けるなど)
- ただし他 5 matchup で平均 -10pp 悪化、 overall **net negative**
- 結論: 「**deck 切替えだけ**」 は不十分。 policy が features を
  deck.csv (我々 Mega Aboma) で訓練済みのため、 deck 入替えで features
  representation が崩れる
- 真の解決は「features に自分の deck-id fingerprint も追加」 だが、
  STATE_DIM 変更 → warm-start 不可 → 大規模再学習が必要

### サマリ (V60 + deck 実験の総括)

| アプローチ | overall | 結論 |
|---|---|---|
| 3-MLP + 我々 deck (現状) | 20.0% / 23.3%@80g | LB 679.6 baseline |
| 3-MLP + V6 deck | 16.7% (20g) | net negative |
| V60 EXT 5500ep | 21.1% (30g) | +0pp、 ノイズ内 |
| V60 EXT2 8500ep | 20.0% | 振動 |
| V60 BIG 128-64 fresh 3000ep | 16.7% | 容量不足解消せず |
| V60 LINVAL fresh 3000ep | 9.4% | tanh は必要 |
| V60 + V6 deck fresh 4000ep | 13.9% | 戦略空間複雑、 収束せず |
| **3-MLP + V6 deck cross-bench** | 16.7% | deck 切替えは net negative |

→ 単純な改造案は全部限界。 **真の本命は PIMC** (search で相手を読む)、
  もしくは **deck-builder agent で新 deck 構築** (Task #107 v4+)。

### PIMC v2: 値関数強化 (2026-06-18)

`train/pimc_agent.py:_prize_delta()` を v2 に拡張:
- prize delta (×100、 dominant) + active HP ratio (×10) +
  bench fill (×2) + active energy count (×1.5)
- 「prize は最重要、 field 状態は marginal」 という重み付け

bench (3 opp × 20g):

  vs Mega Lucario: 1-19 ( 5.0%) ← v1 30% から **-25pp 大幅悪化**
  vs Iono:         3-17 (15.0%) ← v1 10% から +5pp
  vs Crustle Dashi:2-18 (10.0%) ← v1 0% から +10pp
  subtotal:       6-54 (10.0%) ← v1 ~13% から微減

**判定**: matchup 単位の改善悪化が打ち消し合い、 overall 不変。
heuristic value function の改造では本質解決にならない。

### PIMC v3 への方針 (本サイクル末尾の整理)

- 単なる value heuristic 拡張は overall 改善せず (v1=8.3%、 v2=10%)
- 真に必要なのは:
  1. **learned value head** (= NN を value function に使う、 AlphaZero 派生)
  2. **multi-ply rollout** (1-ply の prize delta だけでは情報量が少ない)
  3. **opp deck inference** (相手の手札サンプリングを uniform random から、
     見えた action や discard からの推定に置き換え)

実装優先順:
- (3) opp deck inference: 中規模、 effort vs gain 良し
- (1) learned value head: 既存 V60 EXT policy の value head を流用可能、
  PIMC 結果 + V(s) で argmax
- (2) multi-ply: 高 cost、 後回し

### PIMC v3: V60 EXT value head 統合 — **失敗**

`train/pimc_agent.py:make_v60_value_fn()` を追加、 V60 EXT (5500ep selfplay)
の `.value()` を nn_value_fn として PIMC に注入。 weight を 5/10/30 で sweep:

| weight | subtotal (3 opp, 20g/opp) |
|---|---|
| 5.0 | **1.7%** ← 最弱 |
| 10.0 | 5.0% |
| 30.0 | 6.7% |
| **0 (v2 baseline)** | **10.0%** ← ベスト |

NN value head の統合は **全 weight で baseline 以下**。 原因推察:
1. V60 EXT policy の value head は **selfplay 文脈で訓練済み** = 「自分が
   playing する deck の局面分布」 に偏る
2. PIMC は opp_hand を uniform random でサンプル → search_step で展開した
   obs は selfplay と全く違う局面分布
3. その奇妙な obs に対する V(s) は **noise signal** を発する
4. weight が大きいほど prize delta heuristic を歪める

**結論**: AlphaZero スタイル PIMC + NN value は、 value head の質に強く依存。
我々の V60 EXT は **selfplay 局面分布で偏った value head** を持つので、 PIMC
のような「未知の opp 仮定」 文脈では信頼できない。

### PIMC v4 への方針 (本サイクル末尾)

1. **value head 専用訓練が必要**: PIMC の rollout 文脈 (= search_step の
   obs) で V(s) を学習させる必要。 これは self-distillation 系の手法、
   大規模実装
2. **opp deck inference を先に**: heuristic 限界の中で、 「opp_hand を
   見えた actions から推定する」 ほうが先効果。 PIMC v4 は inference + v2
3. **multi-ply rollout**: 1-ply 後の prize delta は情報量が少なく、 3-5 ply
   進めれば signal が出る可能性。 ただし search_step cost と branching factor
   次第

### PIMC v4: opp deck inference (2026-06-18)

`train/pimc_agent.py:infer_opp_deck()` を追加: 相手の active/bench/discard
で見えた card ID を集めて、 7 種 vendored deck (Iono, Crustle Dashi, Crustle
Wall, Mega Lucario, Dragapult, Aboma, V6) との overlap が最大のものを
opp_deck_assumption として動的に置換。

bench 結果 (5 opp × 20g):

  vs Mega Lucario: 10.0%
  vs Dragapult:    20.0%
  vs Iono:          5.0%
  vs Crustle Dashi: 0.0%
  vs V6:           15.0%
  overall:        10.0% ← v2 baseline と同水準、 noise 内

期待した「相手 deck を当てれば PIMC 精度上がる」 効果は **marginal**。

### PIMC 路線の総括 (4 試行)

| version | feature | subtotal | 結論 |
|---|---|---|---|
| v1 | 1-ply、 prize delta、 fixed Iono opp 仮定 | 8.3% (6 opp、 80g) | baseline、 動作確認 |
| v2 | + field-aware (HP/bench/energy) heuristic | 10% (3 opp、 20g) | 微改善 |
| v3 | + V60 EXT value head | 1.7-6.7% | **失敗** (selfplay 文脈ズレ) |
| v4 | + opp deck inference | 10% (5 opp、 20g) | marginal |

**PIMC 1-ply の天井は 10% 程度** と確認。 これ以上は:
- multi-ply rollout (= search_step 連続 + 終局までシミュ)
- PIMC 文脈で value head 専用訓練 (self-distillation)
- 両方とも実装に複数サイクル必要

### 次サイクル方針

PIMC 路線は **基礎実装完了** で一旦区切る。 1-ply 10% は LB submission
には程遠い (現状 V6 873)。

次の打ち手候補 (優先順):
1. **PIMC を multi-ply 化** (大型実装、 3-5 サイクル)
2. **deck-builder v4** (実 bench fitness で deck を進化、 Task #107 続き)
3. **他の Kaggle kernel を vendor** (まだ未試行: nursrijan、 pilkwang 等)
4. **3-MLP / V60 EXT を LB に再提出** (現状 LB 評価で V6 が落ち着き始めた、
   再評価で順位回復の可能性)

## zoli800 Dragapult tempo-control を vendor (Task #106、 2026-06-18)

`zoli800/top-dragapult-ex-tempo-control-agent` (4 votes) を取り込み:
- `deck_zoli_dragapult.csv` (60 cards、 Dreepy/Drakloak/Dragapult ex 軸)
- `scripts/rule_based_zoli_dragapult.py` (25 KB、 DECK が hardcoded のため
  deck path patch 不要)
- smoke OK: agent callable、 deck head [119, 119, 119, 119, 120]
- 既存 Kiyota Dragapult は 1 energy attack 即発動型、 zoli800 は
  tempo-control 派生

bench は次サイクル (subjects に追加して 80g/opp 計測)。

### zoli800 @ 80g 計測結果

  vs Mega Lucario:    16-63 (20.0%)
  vs Dragapult ex:    14-66 (17.5%) ← mirror も弱い
  vs Iono:            23-57 (28.7%)
  vs Mega Aboma:      15-65 (18.8%)
  vs Crustle Wall:    14-66 (17.5%)
  vs Crustle Dashi:    0-80 ( 0.0%) ← 致命的
  overall:           82-397 (17.1%) ← **最下位**

Kiyota Dragapult (overall 48.1%) と比較して **-31pp** で大幅に弱い。
「tempo control」 はラベルのみで実装が弱い様子。 提出候補から外す。

### 9 subject 最新ランキング @ 80g

| rank | subject | overall |
|---|---|---|
| 1 | CrustleDashi | 67.3% |
| 2 | Iono | 64.0% |
| 3 | RomanrozenV6 | 57.9% |
| 4 | Kojimar | 52.7% |
| 5 | Dragapult (Kiyota) | 48.1% |
| 6 | Lucario (Kiyota) | 46.5% |
| 7 | Abomasnow | 40.0% |
| 8 | CrustleWall (haru) | 36.9% |
| **9** | **ZoliDragapult** | **17.1%** ← 新参 |

### V60 EXT2 (8500ep, lr=1e-4) 結果 — 振動を観測

  vs Mega Lucario:    4-26 (13.3%) ← EXT1 26.7% から **-13.4pp**
  vs Dragapult ex:    4-26 (13.3%) ← EXT1 26.7% から **-13.4pp**
  vs Iono:            5-25 (16.7%) ← +3.4pp
  vs Mega Abomasnow:  9-21 (30.0%) ← 変化なし
  vs Crustle Dashi:   0-30 ( 0.0%) ← **-13.3pp 致命的悪化**
  vs V6:             14-16 (46.7%) ← **+30pp の大幅改善**
  overall:           36-144 (20.0%) ← -1.1pp (ノイズ内)

**policy が学習過程で大きく振動している**。 30g ノイズ floor (±13pp) を
個別 matchup で超えているケース複数 (V6 +30, Lucario -13, Dragapult -13)。

仮説:
- lr=1e-4 でも policy gradient の variance が大きく、 特定 matchup に
  overfit → 別 matchup を忘れる、 を繰り返す
- value baseline tanh が依然 hard matchup で歪んでいる (前述の問題が再現)
- 単一 policy 容量 (64-32) で 6 deck 全部に対応するのは難しい、
  「ensemble of pool-trained」 か 「相手別 sub-policy」 が必要

判断:
- EXT2 を `archive/` に退避、 EXT1 (5500ep) を **作業ベースライン** に維持
- 次サイクル: 別 seed で V60 学習を複数走らせ、 ensemble化を試みる
  (V40 で失敗したが、 features60 で再挑戦)
- ただし「単一 agent 強化」 方針なので、 まず ensemble なしで policy
  容量増 + value baseline 修正を優先

## 🎯 方針転換 (2026-06-18 user 指示)

### 単一 agent への統一

User 指示: 「デッキごとに agent を設けないで 1 つのエージェントを強化する」

- 現状の submission 候補は 4 つ (Iono 762.2 / CrustleDashi 894.2 / V6 897.6 / 3-MLP 679.6)
  だが、これらは全て **deck × rule-based agent のペア**
- 今後の方針: **deck.csv (我々のオリジナル) を提出 deck として固定**、
  そこで動く **学習 policy 1 つ** を継続的に強化していく
- 既存の rule-based agent (Iono / Crustle / Lucario / V6 / kojimar 等) は
  「対戦相手」 (= bench/training opponent) として利用、 自分の提出には使わない
- submission_rule_based_*.tar.gz は **legacy** として `archive/` 行き候補

つまり目指す形:
```
deck.csv (固定 or deck-builder agent が生成)
   ↓
 main.py = 学習 policy (現状 3-MLP、 将来は v60 / v100 / PIMC + NN)
   ↓
 submit
```

### deck の進化

User 指示: 「既存だけでなく対戦評価の上で構築し直せる deck」

- 短期: 外部 Kaggle kernel + 公開 deck list から deck pool 拡充
  (zoli800 Dragapult、 pilkwang Lucario v2、 nursrijan Lucario 等)
- 中期: 自分の deck.csv を「対戦評価」 ベースで進化させる
  (= 既存 deck × 改造案を bench して、 winrate の高い deck を選ぶ GA loop)
- 長期: **deck-building agent** を別途実装
  - 入力: 全カード DB (kaggle_data の EN_Card_Data.csv 等)
  - 出力: 60 枚の deck.csv
  - 中身: super-effective / HP / damage / energy 効率を heuristic か RL で評価

これら 3 段階を順に進める。

### 8 subject 最新ランキング @ 80g

| rank | subject | overall | min (致命弱点) | anti-Crustle |
|---|---|---|---|---|
| 1 | CrustleDashi | 67.3% | **Iono 11.2%** | self (50%) |
| 2 | Iono | 64.0% | Lucario 21.2% | **97.5/92.5%** |
| 3 | RomanrozenV6 | 57.9% | Lucario 43.8% | **51.2/65.0%** (built-in) |
| 4 | Kojimar | 52.7% | Crustle Dashi 21.2% | 30/21.2% |
| 5 | Dragapult | 48.1% | Crustle Dashi 0% | 58.8/0% |
| 6 | Lucario | 46.5% | Crustle 25/22.5% | 25/22.5% |
| 7 | Abomasnow | 40.0% | - | 67.5/16.2% |
| 8 | CrustleWall (haru) | 36.9% | Iono 1.2% | self/10% |

### rule_based(Iono) の特徴

- **Crustle 系を完封**: Crustle Wall 97.5% / Crustle Dashimaki 92.5%
  (= LB の dashimaki 派 AM, sbite0138, PavelLiashkov を 90%+ で殺せる)
- Aboma 68.8%, Iono mirror 57.5%, Dragapult 46.2% で堅実
- **唯一の弱点は Mega Lucario @ 21.2%** (Wattrel/Lightning は Fighting 弱点なし
  ではあるが、Mega Lucario の打点が高すぎて押し切られる)
- overall 64.0% は Lucario より **+17.5pp**、3-MLP より **+40.7pp**

### Dragapult との比較

- Dragapult @ 48.1% は Crustle Dashimaki に **0% 全敗** (致命的)
- Iono @ 64.0% は弱点が Mega Lucario 単一なので分散が低い
- LB の deck 分布が読めない以上、**全 matchup で 21% を下回らない Iono の方が安全**

### 次サイクル予定

1. `main_rule_based_iono.py` + `make_submission_rule_based_iono.sh` を構築
2. `submission_rule_based_iono.tar.gz` を build & sandbox verify
3. user に「Iono 切替で submit?」を問う (LB 666.3 → 期待大幅向上)

候補比較メモは `train/` には触らず、`submission_rule_based.tar.gz` (Lucario 版)
と並ぶ第 2 候補として位置づける。

### Iono submission build 完了 (2026-06-18)

`main_rule_based_iono.py` と `make_submission_rule_based_iono.sh` を作成、
`submission_rule_based_iono.tar.gz` (1.06MB) を build & sandbox verify 完了。
deck = `[265, 265, 265, ...]` (Iono deck head) で正常起動を確認。

### 現在 user 判断待ちの提出候補 3 つ

| 候補 | tar.gz | overall (lab 80g) | deck |
|---|---|---|---|
| 現状 (3-MLP) | submission.tar.gz | 23.3% (= LB 666.3) | 我々のオリジナル |
| Lucario | submission_rule_based.tar.gz | 46.5% | deck_mega_lucario.csv |
| **Iono** | **submission_rule_based_iono.tar.gz** | **64.0%** | deck_iono.csv |

submit コマンド (どちらか選択):

```bash
# Iono 案 (最有力、lab 64.0%)
.venv/bin/kaggle competitions submit -c pokemon-tcg-ai-battle \
    -f submission_rule_based_iono.tar.gz \
    -m "Switch to rule-based Iono (lab 64.0%, Crustle 97.5/92.5%)"

# Lucario 案 (lab 46.5%)
.venv/bin/kaggle competitions submit -c pokemon-tcg-ai-battle \
    -f submission_rule_based.tar.gz \
    -m "Switch to rule-based Mega Lucario (lab 46.5%)"
```

## 現状サマリ (2026-06-18 evening)

### Submission 状況
- 最新 LB スコア: **666.3** (53778627, 3-MLP)
- 次回提出は Iono ft メンバー追加版を 80g 検証してから判断

### Crustle 対策の総括 (4 連敗)
| 試行 | 手法 | solo (40g) | 結果 |
|---|---|---|---|
| seed=7 | random init 2000ep | 未計測 | 4-MLP @ Crustle 6.2% (-17.6pp) |
| seed=11 crustle | random init 2000ep targeted | 未計測 | 4-MLP @ Crustle 10.0% (-13.8pp) |
| seed=100 ft | warm-start 1000ep pure lr=3e-4 | 12.5% (元 15.0%) | archive |
| seed=100 mix | warm-start 1000ep mixed lr=1e-4 | 5.0% (元 15.0%) | archive |

→ pure REINFORCE では Crustle dashimaki 改善は実現できない。
方向転換が必要 (Crustle 検出切替 / value baseline 正規化 / PIMC)。

### 学習インフラ (積み上げ済み)
- `train/mlp_train.py --opponent <module_name>`: rule-based 相手の targeted self-play
- `train/mlp_train.py --opponent-prob <0..1>`: mixed-mode self-play
- 学習履歴は `train/metrics_*.json` に保存、過去 policy は `train/archive/`

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
