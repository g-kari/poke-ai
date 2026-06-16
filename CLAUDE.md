# CLAUDE.md

開発時に Claude Code が読む補助メモ。プロジェクト固有の事情と"知っておかないと事故る"罠を集約。

## プロジェクト概要

Kaggle コンペ **「ポケモンカードゲーム AI Battle Challenge (PTCGABC / ポケカABC)」** のシミュレーション部門への提出エージェント。

- 公式: https://ptcg-abc.pokemon.co.jp/
- Kaggle: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle
- 締切: シミュ部門 2026-08-17, ストラテジー部門 2026-09-14
- 提出形式: `agent(obs_dict: dict) -> list[int]` を含む `agent.py`

`HANDOVER.md`(別途リポジトリ外で共有)に元の設計判断と歴史が、`NOTES.md` に実機調査ログがある。

## 開発フロー

```bash
# スモークテスト (~3秒 / 8試合)
python3 selfplay_test.py 4

# 学習 (~3分 / 500ep)
python3 -m train.reinforce --episodes 500 --lr 0.05 --metrics-out train/metrics.json

# 学習済み重みは train/policy.npz に保存され、agent.py が起動時に自動ロード
```

## ファイル構成

```
agent.py              # Kaggle 提出エントリ。policy.npz があれば使う / 無ければ engine 順
selfplay_test.py      # vs random ベンチ
train/
  features.py         # state 24-d / option 18-d 特徴
  policy.py           # numpy 線形ポリシー + .npz 保存
  reinforce.py        # 自己対戦 REINFORCE ループ
  policy.npz          # 学習済み重み (リポジトリにコミットしている)
  metrics_*.json      # 学習ログ
NOTES.md              # 実機調査の VERIFY 結果 + obs スキーマ + OptionType 列挙
```

## 重要な前提 ("HANDOVER との差分")

HANDOVER は `cabt.api` という Python wrapper を前提に書かれているが、
**pip 配布版 (`kaggle-environments==1.30.1`) には存在しない**。実機は:

- 入口: `kaggle_environments.envs.cabt.cg.game` の `battle_start/select/finish`
- C 本体: `kaggle_environments.envs.cabt.cg.sim.lib` (libcg.so の ctypes wrap)
- `obs` は plain dict (クラスではない)
- `all_card_data() / search_begin() / ...` は **Python ラッパー無し**。C 関数 (`AllCard`, `SearchBegin` 等) は libcg.so に存在するが ABI 未公開。

→ PIMC / IS-MCTS は当面凍結。**学習ポリシー路線** (`train/`) で進める。

## obs スキーマ早見

```python
obs = {
  "select": None | {
    "type": int, "context": int,
    "minCount": int, "maxCount": int,
    "option": [{"type": OptionType, ...}, ...],
    "deck": list|None, "remainEnergyCost": int, "remainDamageCounter": int,
    "contextCard": dict|None, "effect": dict|None,
  },
  "logs": list[dict],
  "current": None | {
    "turn": int, "turnActionCount": int,
    "yourIndex": 0|1, "firstPlayer": -1|0|1,
    "result": -1|0|1|2,  # -1=in progress, 0/1=winner, 2=draw
    "players": [
      {"active": list, "bench": list, "benchMax": 5,
       "deckCount": int, "discard": list,
       "prize": list, "handCount": int,
       "hand": list|None,  # opponent's is None
       "poisoned": bool, "burned": bool, "asleep": bool,
       "paralyzed": bool, "confused": bool},
      ...
    ],
  },
  "search_begin_input": str,  # opaque ~80-char ASCII (engine state blob)
}
```

### MAIN OptionType (実機観測値)

```
 7  PLAY     {index}
 8  ATTACH   {area, index, inPlayArea, inPlayIndex}
 9  EVOLVE   {area, index, inPlayArea, inPlayIndex}
13  ATTACK   {attackId}
14  END      {}
```

### 戻り値の形

- 単一選択 (`maxCount == 1`): `[i]` を返す
- 複数選択: `minCount..maxCount` 個の **インデックス配列**
- `select is None`: 初期デッキ提出 → **60 枚のカード ID 配列を返す**

## 罠・既知の地雷

1. **litellm import panic**: `from kaggle_environments import make` の前に
   `sys.modules.setdefault("litellm", type(sys)("litellm"))` を入れないと、
   werewolf env のロード時に `cryptography` の Rust 拡張が panic する。
   `selfplay_test.py` と `train/reinforce.py` 冒頭で対処済み。
2. **`random` の `maxCount=0`**: `random.sample(range(n), 0)` は OK だが、
   `maxCount` を超える k を渡さないこと。
3. **対戦エンジンの option 順序は意外と賢い**: 何もしない `list(range(maxCount))`
   ベースラインが random を 7-1 で倒す。学習時はこの prior に勝つ必要がある。
4. **`obs["current"]` が None** のことがある (デッキ提出フェーズ等)。`agent()` は
   先頭で `select is None` を見て早期 return している。
5. **学習中の例外で C 側ハンドルが orphan 化**することがある。`Battle.battle_ptr`
   が None でないまま `battle_start` を呼ぶと未定義動作。`make("cabt")` を毎回
   呼べばリセットされる (`interpreter` 関数内で処理)。

## コミット規約

- 1 機能 1 コミット。コミット message は英語 1 行で意図を書く。
- `train/policy.npz` はコミット対象 (Kaggle 提出時に必要)。
- `train/metrics_*.json` もコミット (学習履歴として残す)。
- それ以外の `*.npz` は `.gitignore` で除外。

## ブランチ

- `main` がデフォルト。
- 開発は `claude/sleepy-franklin-jse7on` (もしくは新しい `claude/...` ブランチ) で行い PR。
- HANDOVER 由来の元実装 (`agent.py` 骨組み) からは大きく逸脱しており、
  必要なら `NOTES.md` の "VERIFY answers" を先頭から読むこと。

## 次の打ち手 (優先順)

1. **長め自己対戦** で学習を積む (`--episodes 2000` で勝率カーブを取る)
2. **特徴量追加**: カード ID 埋め込み、active Pokemon の attack 候補
3. **PyTorch + PPO**: 線形ポリシーの限界を超えるとき
4. **SearchBegin C-ABI 解析**: PIMC 路線復活 (難)
5. **デッキ差し替え**: `agent.py:DECK` をコンペ配布の legal 60 枚に
6. **ストラテジー部門レポート** (締切 9/14)

## 環境メモ

- Python 3.11
- `kaggle-environments==1.30.1` で `cabt` env 同梱を確認
- GPU 無し、CPU で numpy 線形ポリシー。500ep ≈ 3分 / 2000ep ≈ 12分
- リモート ephemeral コンテナ前提 (work in progress は push しないと消える)
