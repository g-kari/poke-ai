"""公式 Top episodes (6/16 dataset) の簡易 EDA。

目的: LB 上位プレイヤーの観測リプレイから、 勝者側で見えたカード ID を
集計し、 現状の deck.csv の妥当性を評価する。

第 1 段階の出力 (= docs/EPISODES_EDA_RESULT.md):
- 勝者側で見えたカード ID 上位 60 件 (= 暫定 deck 案)
- 我々の deck.csv カードとの diff
- チーム別の勝率
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EP_DIR = ROOT / "kaggle_data/episodes_2026-06-16"
N_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def extract_visible_cards(obs: dict, player_idx: int) -> list[int]:
    """obs['current']['players'][player_idx] から見える全カード ID を抽出。"""
    cur = obs.get("current")
    if cur is None:
        return []
    players = cur.get("players") or []
    if player_idx >= len(players):
        return []
    me = players[player_idx]
    ids: list[int] = []
    for area in ("active", "bench", "discard", "hand"):
        for card in me.get(area, []) or []:
            if isinstance(card, dict):
                cid = card.get("cardId") or card.get("id")
                if isinstance(cid, int):
                    ids.append(cid)
                elif isinstance(cid, str) and cid.isdigit():
                    ids.append(int(cid))
            elif isinstance(card, int):
                ids.append(card)
    return ids


def analyze_episode(path: Path) -> dict:
    with open(path) as f:
        ep = json.load(f)
    rewards = ep.get("rewards") or [0, 0]
    if rewards[0] == 1:
        winner = 0
    elif rewards[1] == 1:
        winner = 1
    else:
        winner = None
    teams = ep.get("info", {}).get("TeamNames") or ["?", "?"]
    winner_cards = Counter()
    loser_cards = Counter()
    if winner is not None:
        loser = 1 - winner
        for step in ep.get("steps") or []:
            for pidx in (winner, loser):
                if pidx >= len(step):
                    continue
                obs = step[pidx].get("observation") or {}
                ids = extract_visible_cards(obs, pidx)
                if pidx == winner:
                    winner_cards.update(ids)
                else:
                    loser_cards.update(ids)
    return {
        "winner": winner,
        "winner_team": teams[winner] if winner is not None else None,
        "loser_team": teams[1 - winner] if winner is not None else None,
        "winner_cards": winner_cards,
        "loser_cards": loser_cards,
        "num_steps": len(ep.get("steps") or []),
    }


def main() -> None:
    paths = sorted(EP_DIR.glob("*.json"))[:N_SAMPLE]
    print(f"=== EDA: {len(paths)} episodes from {EP_DIR.name} ===")

    team_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"games": 0, "wins": 0, "losses": 0}
    )
    all_winner_cards: Counter = Counter()
    all_loser_cards: Counter = Counter()

    for i, p in enumerate(paths):
        try:
            r = analyze_episode(p)
        except Exception as e:
            print(f"  [{i}] {p.name}: ERROR {e}", file=sys.stderr)
            continue
        if r["winner"] is None:
            continue
        team_stats[r["winner_team"]]["games"] += 1
        team_stats[r["winner_team"]]["wins"] += 1
        team_stats[r["loser_team"]]["games"] += 1
        team_stats[r["loser_team"]]["losses"] += 1
        all_winner_cards.update(r["winner_cards"])
        all_loser_cards.update(r["loser_cards"])
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(paths)}", flush=True)

    print("\n=== TEAM RANKING (= LB top of 6/16, top 20 by wins) ===")
    ranked = sorted(
        team_stats.items(),
        key=lambda kv: (kv[1]["wins"], -kv[1]["losses"]),
        reverse=True,
    )
    for team, s in ranked[:20]:
        wr = s["wins"] / max(1, s["games"])
        print(
            f"  {team:30s}  games={s['games']:3d}  W={s['wins']:3d}  L={s['losses']:3d}  WR={wr:.1%}"
        )

    print("\n=== WINNER-SIDE TOP CARDS (top 60) ===")
    for card_id, cnt in all_winner_cards.most_common(60):
        loser_cnt = all_loser_cards.get(card_id, 0)
        delta = cnt - loser_cnt
        print(
            f"  cardId={card_id:5d}  winner_obs={cnt:6d}  loser_obs={loser_cnt:6d}  delta={delta:+6d}"
        )

    print("\n=== WINNER - LOSER DELTA (top 30 winner-skewed) ===")
    delta_counter: Counter = Counter()
    all_ids = set(all_winner_cards) | set(all_loser_cards)
    for cid in all_ids:
        delta_counter[cid] = all_winner_cards.get(cid, 0) - all_loser_cards.get(cid, 0)
    for card_id, delta in delta_counter.most_common(30):
        print(f"  cardId={card_id:5d}  delta={delta:+6d}")

    print("\n=== Summary ===")
    print(f"  total episodes processed: {sum(1 for p in paths)}")
    print(f"  unique cards seen (winner side): {len(all_winner_cards)}")
    print(f"  unique teams: {len(team_stats)}")


if __name__ == "__main__":
    main()
