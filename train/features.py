"""obs_dict -> feature vectors.

Two extractors:
  state_features(obs)      : per-step features describing the board state
                             (independent of which option is being scored).
  option_features(opt, obs): features describing a candidate option in context.

The policy net scores each option as
  logit_i = w_state . state_features(obs) + w_opt . option_features(opt_i, obs)
so the model has constant size regardless of how many options are offered.
"""

from __future__ import annotations

import numpy as np

STATE_DIM = 24
OPTION_DIM = 18


def state_features(obs: dict) -> np.ndarray:
    """Compact per-decision board summary, shape (STATE_DIM,)."""
    f = np.zeros(STATE_DIM, dtype=np.float32)
    cur = obs.get("current")
    if cur is None:
        return f
    you = cur["yourIndex"]
    me = cur["players"][you]
    opp = cur["players"][1 - you]
    f[0] = cur.get("turn", 0) / 30.0
    f[1] = cur.get("turnActionCount", 0) / 10.0
    f[2] = float(cur.get("supporterPlayed", False))
    f[3] = float(cur.get("stadiumPlayed", False))
    f[4] = cur.get("energyAttached", 0)
    f[5] = float(cur.get("retreated", False))
    f[6] = me.get("handCount", 0) / 10.0
    f[7] = me.get("deckCount", 0) / 60.0
    f[8] = len(me.get("prize", [])) / 6.0
    f[9] = len(me.get("discard", [])) / 60.0
    f[10] = len(me.get("bench", [])) / 5.0
    f[11] = 1.0 if me.get("active") else 0.0
    f[12] = opp.get("handCount", 0) / 10.0
    f[13] = opp.get("deckCount", 0) / 60.0
    f[14] = len(opp.get("prize", [])) / 6.0
    f[15] = len(opp.get("discard", [])) / 60.0
    f[16] = len(opp.get("bench", [])) / 5.0
    f[17] = 1.0 if opp.get("active") else 0.0
    # Active HP totals
    f[18] = _hp_total(me.get("active")) / 200.0
    f[19] = _hp_total(opp.get("active")) / 200.0
    f[20] = _hp_total(me.get("bench")) / 1000.0
    f[21] = _hp_total(opp.get("bench")) / 1000.0
    # Status
    f[22] = float(me.get("poisoned") or me.get("burned") or me.get("asleep")
                  or me.get("paralyzed") or me.get("confused"))
    f[23] = float(opp.get("poisoned") or opp.get("burned") or opp.get("asleep")
                  or opp.get("paralyzed") or opp.get("confused"))
    return f


def option_features(opt: dict, obs: dict, sel: dict) -> np.ndarray:
    """Per-option features, shape (OPTION_DIM,). Mostly one-hot OptionType."""
    f = np.zeros(OPTION_DIM, dtype=np.float32)
    t = opt.get("type", -1)
    # OptionType one-hot for the values we know about.
    type_slots = {0: 0, 1: 1, 2: 2, 3: 3, 7: 4, 8: 5, 9: 6, 13: 7, 14: 8}
    if t in type_slots:
        f[type_slots[t]] = 1.0
    f[9] = (opt.get("index") or 0) / 60.0
    f[10] = (opt.get("inPlayIndex") or 0) / 6.0
    f[11] = (opt.get("number") or 0) / 10.0
    # Area one-hot bucketed
    area = opt.get("area")
    if area is not None:
        f[12] = float(area == 2)   # HAND
        f[13] = float(area == 4)   # ACTIVE
        f[14] = float(area == 5)   # BENCH
        f[15] = float(area == 3)   # DISCARD
    # Select context
    f[16] = (sel.get("type") or 0) / 12.0
    f[17] = (sel.get("context") or 0) / 50.0
    return f


def _hp_total(pokes) -> float:
    if not pokes:
        return 0.0
    return float(sum((p.get("hp") or 0) for p in pokes if p))
