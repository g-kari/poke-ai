# Engineering notes (PTCGABC / cabt)

Living doc of what the actual installed env exposes, paired with the HANDOVER
items that need correcting and the path to a learned policy.

## VERIFY answers (HANDOVER §5)

The HANDOVER was written against the docs at `https://matsuoinstitute.github.io/cabt/`
(which describe a pure-Python `cabt.api` wrapper). The version shipped via
`pip install kaggle-environments==1.30.1` does **not** expose that wrapper.
Findings from an actual self-play run on this image:

1. **Import path.** The Python entry points are
   `kaggle_environments.envs.cabt.cg.game` (`battle_start`, `battle_select`,
   `battle_finish`, `visualize_data`) and `...cg.sim` (`lib`, `Battle`,
   `StartData`, `SerialData`). There is no top-level `cabt` package, no
   `cabt.api`, and no Python wrapper for `all_card_data() / all_attack() /
   to_observation_class() / search_begin() / search_step() / search_end() /
   search_release()`. The C symbols exist in `libcg.so` (`AllCard`,
   `AllAttack`, `SearchBegin`, `SearchStep`, `SearchEnd`, `SearchRelease`)
   but the ABI is not documented and `AllAttack()` with no args returns
   `count=0`, so reverse-engineering it is risky without official docs.
2. **Terminal & value.** `obs["current"]["result"]` is `-1` while the game
   continues, `0` if P0 wins, `1` if P1 wins, `2` for draw. `env.steps[-1][i].reward`
   exposes the same as `+1 / 0 / -1` to the agent function output.
3. **Search rewind.** Not testable without a Python `SearchBegin` wrapper.
   Until one is wired in, **drop the PIMC layer entirely** and use a
   learned policy on the raw observation. The notes below describe the
   training path.
4. **`search_begin_input`.** Plain ASCII string ~80 chars at the opening
   step (`'AGEAjD/...=='`), produced by `GetBattleData` from the C side.
   It is the engine's serialized state buffer — opaque to Python without
   a `SearchBegin` wrapper.
5. **`legal_pool`.** The bundled sample deck in
   `kaggle_environments.envs.cabt.cabt.deck` is 60 IDs (mostly id `3`,
   plus a few in the 700-1300 range). The actual contest legal-card list
   has to come from the Kaggle competition data tab — not from this
   package.
6. **`deckCount` vs `len(deck)`.** `deckCount` always reflects the visible
   remaining cards; the `hand`/`deck`/`prize` arrays are exactly that long.

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

### MAIN OptionType values (empirical)

```
 7  PLAY     {"index": int}
 8  ATTACH   {"area": 2 (HAND), "index", "inPlayArea": 4|5, "inPlayIndex"}
 9  EVOLVE   {"area": 2, "index", "inPlayArea", "inPlayIndex"}
13  ATTACK   {"attackId": int}
14  END      {}
```

`ABILITY / DISCARD / RETREAT` were not observed in mirror baseline play but
should also be supported by the engine.

## Files

```
agent.py              # Kaggle submission entry point. Loads train/policy.npz
                      # if present, otherwise uses the "engine ordering"
                      # baseline (still beats random ~7-1).
selfplay_test.py      # python3 selfplay_test.py [N]  → benchmark vs random
train/
  features.py         # obs -> state/option feature vectors
  policy.py           # Linear policy (numpy), save/load .npz
  reinforce.py        # python3 -m train.reinforce --episodes N
  policy.npz          # trained weights (created by training)
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
5. **Search.** If a Python `SearchBegin` wrapper becomes available (or
   we figure out the C ABI from official docs), wrap the trained policy
   as the rollout policy inside MCTS / IS-MCTS as the HANDOVER originally
   envisioned.

## Open items

- `_try_load_policy()` silently swallows exceptions to keep the Kaggle
  submission robust. Add a CI hook that loads weights on a clean checkout
  so we catch breakage before submitting.
- The submission deck is still the env's sample deck. Replace with the
  competition-legal 60-card list once obtained.
- `NUMBER (type=0)` selects (e.g. for "draw N cards" prompts) are routed
  through the same policy; we should special-case them once we see one
  in the wild during a non-mirror match.
