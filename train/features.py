"""obs_dict -> feature vectors.

Two extractors:
  state_features(obs)         : per-step features (constant w.r.t. options).
  option_features(opt, obs, sel): per-candidate features.

Logit for option i = w_state . state_features + w_opt . option_features.
This keeps the policy size independent of how many options the engine offers.
"""

from __future__ import annotations

import numpy as np

STATE_DIM = 36
OPTION_DIM = 36

# Card ID hash buckets — cheap proxy for "this is the same card I saw last time"
# without needing the C-side AllCard table. Increase if collisions become a
# problem in practice.
_ID_BUCKETS = 8
_ATTACK_BUCKETS = 8


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
    f[18] = _hp_total(me.get("active")) / 200.0
    f[19] = _hp_total(opp.get("active")) / 200.0
    f[20] = _hp_total(me.get("bench")) / 1000.0
    f[21] = _hp_total(opp.get("bench")) / 1000.0
    f[22] = float(any((me.get(s) for s in ("poisoned", "burned", "asleep",
                                            "paralyzed", "confused"))))
    f[23] = float(any((opp.get(s) for s in ("poisoned", "burned", "asleep",
                                             "paralyzed", "confused"))))

    # --- richer per-Pokemon stats ---
    my_act = (me.get("active") or [None])[0]
    opp_act = (opp.get("active") or [None])[0]
    f[24] = _hp_ratio(my_act)
    f[25] = _energy_count(my_act) / 6.0
    f[26] = float(my_act.get("appearThisTurn", False)) if my_act else 0.0
    f[27] = len(my_act.get("preEvolution", [])) / 2.0 if my_act else 0.0
    f[28] = _hp_ratio(opp_act)
    f[29] = _energy_count(opp_act) / 6.0
    f[30] = len(opp_act.get("preEvolution", [])) / 2.0 if opp_act else 0.0

    # Bench aggregates.
    bench = [p for p in (me.get("bench") or []) if p]
    f[31] = (sum(_hp_ratio(p) for p in bench) / max(1, len(bench))) if bench else 0.0
    f[32] = sum(_energy_count(p) for p in bench) / 10.0
    f[33] = len(bench) / 5.0  # filled bench slots
    # Prize differential (positive = we are ahead in prize-taking).
    f[34] = (len(opp.get("prize", [])) - len(me.get("prize", []))) / 6.0
    # Hand-card-id histogram folded into a single scalar (cheap diversity proxy).
    hand = me.get("hand") or []
    if hand:
        ids = {c["id"] for c in hand}
        f[35] = len(ids) / max(1, len(hand))
    return f


def option_features(opt: dict, obs: dict, sel: dict) -> np.ndarray:
    """Per-option features, shape (OPTION_DIM,)."""
    f = np.zeros(OPTION_DIM, dtype=np.float32)
    t = opt.get("type", -1)
    type_slots = {0: 0, 1: 1, 2: 2, 3: 3, 7: 4, 8: 5, 9: 6, 13: 7, 14: 8}
    if t in type_slots:
        f[type_slots[t]] = 1.0
    f[9] = (opt.get("index") or 0) / 60.0
    f[10] = (opt.get("inPlayIndex") or 0) / 6.0
    f[11] = (opt.get("number") or 0) / 10.0
    area = opt.get("area")
    if area is not None:
        f[12] = float(area == 2)   # HAND
        f[13] = float(area == 4)   # ACTIVE
        f[14] = float(area == 5)   # BENCH
        f[15] = float(area == 3)   # DISCARD
    f[16] = (sel.get("type") or 0) / 12.0
    f[17] = (sel.get("context") or 0) / 50.0

    # --- card- and target-aware features ---
    cur = obs.get("current")
    if cur is None:
        return f
    you = cur["yourIndex"]
    me = cur["players"][you]

    # Card-id hash buckets for the card this option references (when it lives
    # in our hand). Lets the policy learn "this specific card is good to play
    # in this state" without needing the engine's full card DB.
    card_id = _card_id_referenced(opt, me)
    if card_id is not None:
        f[18 + (card_id % _ID_BUCKETS)] = 1.0

    # ATTACK-only: hash the attackId.
    if t == 13:
        a_id = opt.get("attackId")
        if a_id is not None:
            f[18 + _ID_BUCKETS + (a_id % _ATTACK_BUCKETS)] = 1.0

    # Target Pokemon stats for ATTACH/EVOLVE (inPlayArea/inPlayIndex).
    target = _resolve_in_play(opt, me)
    if target is not None:
        # Reuse the trailing two slots for HP-ratio + energy count of the
        # target. (Indices 30, 31 in the option vector.)
        f[OPTION_DIM - 2] = _hp_ratio(target)
        f[OPTION_DIM - 1] = _energy_count(target) / 6.0
    return f


def _hp_total(pokes) -> float:
    if not pokes:
        return 0.0
    return float(sum((p.get("hp") or 0) for p in pokes if p))


def _hp_ratio(p) -> float:
    if not p:
        return 0.0
    mx = p.get("maxHp") or 0
    return (p.get("hp") or 0) / mx if mx > 0 else 0.0


def _energy_count(p) -> int:
    if not p:
        return 0
    return len(p.get("energies") or [])


def _card_id_referenced(opt: dict, me: dict) -> int | None:
    """If the option references a card by its hand index, return that card's
    id. Returns None when there's no straightforward mapping (e.g., END,
    pure ATTACK)."""
    area = opt.get("area")
    idx = opt.get("index")
    if area == 2 and idx is not None:   # HAND
        hand = me.get("hand") or []
        if 0 <= idx < len(hand):
            return hand[idx].get("id")
    return None


def _resolve_in_play(opt: dict, me: dict) -> dict | None:
    """Resolve the in-play Pokemon an option targets via inPlayArea/inPlayIndex."""
    ipa = opt.get("inPlayArea")
    ipi = opt.get("inPlayIndex")
    if ipa is None or ipi is None:
        return None
    if ipa == 4:  # ACTIVE
        actives = me.get("active") or []
        return actives[ipi] if 0 <= ipi < len(actives) else None
    if ipa == 5:  # BENCH
        bench = me.get("bench") or []
        return bench[ipi] if 0 <= ipi < len(bench) else None
    return None
