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

1. **Search (UNFROZEN 2026-06-17; 1-ply scaffold landed but not yet a win).**
   `cg.api.search_begin/step/release` is wired through `train/pimc.py` and
   gated by `POKEAI_PIMC=1` in `main.py`. With either the heuristic value
   or the trained linear value head, PIMC-ON UNDERPERFORMS the engine-prior
   linear baseline. Diagnostic benches (40 games each):
   - PIMC-ON vs PIMC-OFF mirror match: 10-30 (25%)
   - PIMC-ON vs first_agent (engine-prior baseline): 15-25 (37.5%)
   - PIMC-ON vs random: 75% (PIMC-OFF wins 92.5% on the same setup)

   Suspected reasons (un-fixed): the value head is trained on MAIN-state
   features and gets evaluated on the children of search_step which can
   land at CARD / ENERGY sub-selects where the head extrapolates; the
   single-world mirror-deck sample biases the search; the value head
   signal is too low-magnitude to override the engine-order bias. Path
   forward: multi-world sampling, PyTorch MLP value head, or replacing
   the entire flow with rollout-to-terminal IS-MCTS.
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
