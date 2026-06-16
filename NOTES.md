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
"drop entirely". `cg.api` exposes:

- `Observation / SelectData / Option / State / PlayerState / Pokemon / Card / SearchState / Skill / CardData / Attack` (dataclasses)
- `OptionType (0-16)`, `SelectType (0-10)`, `SelectContext`, `AreaType`, `EnergyType (0-11)`, `CardType (0-6)`, `SpecialConditionType (0-4)`, `LogType`
- `all_card_data() -> list[CardData]`
- `all_attack() -> list[Attack]`
- `to_observation_class(obs_dict) -> Observation`
- `search_begin(agent_observation, your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active, manual_coin=False) -> SearchState`
- `search_step(search_id, select) -> SearchState`
- `search_end()`, `search_release(search_id)`

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

```bash
# (Optional) burn-in: 200 self-play games, ~80 s on this image.
python3 -m train.reinforce --episodes 200 --lr 0.05

# Re-benchmark with weights loaded.
python3 selfplay_test.py 8
```

The current setup is intentionally tiny: numpy-only, linear policy on a
24-dim state vector + 18-dim option vector, REINFORCE with the terminal
reward. It beats `random_agent` 14-2 after 40 self-play episodes.

### Upgrade path

1. **Better features.** Add card-id embeddings (look up `Pokemon.id`,
   energy types, attack costs) and tile-encoded counts (active/bench HP
   per slot).
2. **Bigger model.** Drop in a small MLP (PyTorch) once the feature dim
   stops being the bottleneck. `pip install torch --user`.
3. **PPO + value baseline.** Replace `reinforce_update` with a clipped
   PPO objective and a value head; that removes the high-variance terminal
   reward signal that REINFORCE suffers from.
4. **Self-play league.** Keep a rotating snapshot of past policies and
   train against the league instead of only the current policy — prevents
   cycle-collapse where the agent over-fits to its own quirks.
5. **Search (UNFROZEN 2026-06-17).** `cg.api.search_begin/step/release` is
   available — wrap the trained linear policy as the rollout policy inside
   IS-MCTS. For the hidden-info sampler, draw `opponent_hand` from cards
   we haven't seen yet (deck minus visible cards minus our predicted hand).

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
