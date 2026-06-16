# CLAUDE.md

開発時に Claude Code が読む補助メモ。プロジェクト固有の事情と"知っておかないと事故る"罠を集約。

## プロジェクト概要

Kaggle コンペ **「ポケモンカードゲーム AI Battle Challenge (PTCGABC / ポケカABC)」** のシミュレーション部門への提出エージェント。

- 公式: https://ptcg-abc.pokemon.co.jp/
- Kaggle: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle
- 公式 API ドキュメント: https://matsuoinstitute.github.io/cabt/
- 締切:
  - 2026-08-09 応募/チーム合併
  - 2026-08-16 最終提出 (シミュ)
  - 2026-08-17 ～ 2026-08-31 評価期間 (ランキング確定)
  - 2026-09-14 ストラテジー部門レポート
- 1 日提出枠: 最大 5 件 (最新 2 件のみ追跡)
- 評価方式: TrueSkill 風 μ₀=600、勝敗で μ 更新、不確実性 σ 縮小
- 提出形式: **`.tar.gz` (`tar -czvf submission.tar.gz *`)、ルート直下に `main.py` (ネスト不可) と `deck.csv`**

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
main.py               # Kaggle 提出エントリ。agent(obs)→list[int]。policy.npz があれば使う / 無ければ engine 順
agent.py              # ローカル開発用の旧エントリ (selfplay_test 等が参照)。提出には使われない
deck.csv              # 提出デッキ 60 行 (card ID per line)
cg/                   # 公式サンプル同梱の engine ラッパー
  __init__.py
  api.py              # Observation / Option / SearchState 等 dataclass + 全関数ラッパー
  game.py             # battle_start / battle_select / battle_finish
  sim.py              # ctypes 経由の libcg/cg.dll バインディング
  utils.py            # JSON ↔ dataclass 変換
  libcg.so            # Linux 用 C 本体
  cg.dll              # Windows 用 C 本体
selfplay_test.py      # vs random ベンチ
make_submission.sh    # .tar.gz バンドル生成 (./make_submission.sh -> submission.tar.gz)
train/
  features.py         # state 36-d / option 36-d 特徴 (richer feature 化済)
  policy.py           # numpy 線形ポリシー + .npz 保存
  reinforce.py        # 自己対戦 REINFORCE ループ
  policy.npz          # 学習済み重み (リポジトリにコミットしている)
  metrics_*.json      # 学習ログ
kaggle_data/          # Kaggle 配布物展開先 (sample_submission, *_Card_Data.csv 等)
pokemon-tcg-ai-battle.zip  # Kaggle CLI で DL した配布 zip (PDF カードリスト 300MB 含む)
NOTES.md              # 実機調査の VERIFY 結果 + obs スキーマ + OptionType 列挙
```

## 重要な前提 ("HANDOVER との差分") — 2026-06-17 更新

**朗報: 公式が `cg/api.py` (26KB) を sample_submission に同梱しており、
HANDOVER が前提とした Python ラッパーは全て揃っていた。** pip の
`kaggle-environments` には無いだけ。`PIMC / IS-MCTS は凍結` は誤りで、
`cg/api.py` を提出に同梱して以下の関数が使える:

- `search_begin(agent_observation, your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active, manual_coin=False) -> SearchState`
- `search_step(search_id, select) -> SearchState`
- `search_end()`, `search_release(search_id)`
- `all_card_data() -> list[CardData]`, `all_attack() -> list[Attack]`
- `to_observation_class(obs) -> Observation` (dict → dataclass)
- `battle_start/select/finish` も `cg.game` 経由

これで **PIMC / IS-MCTS 路線が解凍**。`opponent_hand` 等を予測してサンプリング
→ `search_step` で展開 → 終局報酬から ROOT の Q を見積もる、が素直に書ける。

なお、**ローカル開発で `kaggle_environments.envs.cabt` を使う際の旧ノートは
`agent.py` の冒頭 docstring に残っているが、提出ランタイムでは `cg/` の方を
信用すること**(提出には `kaggle_environments` の保証無し)。

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

### OptionType 一覧 (`cg.api.OptionType` 公式定義より)

```
 0  NUMBER           {number}                                          数値選択
 1  YES              {}                                                Yes
 2  NO               {}                                                No
 3  CARD             {area, index, playerIndex}                        指定領域からカード選択
 4  TOOL_CARD        {area, index, playerIndex, toolIndex}             付与ツール選択
 5  ENERGY_CARD      {area, index, playerIndex, energyIndex}           付与エネルギーカード選択
 6  ENERGY           {area, index, playerIndex, energyIndex, count}    エネルギー単位の選択
 7  PLAY             {index}                                           手札からのプレイ
 8  ATTACH           {area, index, inPlayArea, inPlayIndex}            エネルギー手付け
 9  EVOLVE           {area, index, inPlayArea, inPlayIndex}            進化
10  ABILITY          {area, index}                                     特性発動
11  DISCARD          {area, index}                                     場のカードをトラッシュ
12  RETREAT          {}                                                逃げる
13  ATTACK           {attackId}                                        ワザ宣言
14  END              {}                                                ターン終了
15  SKILL            {cardId, serial}                                  カードの効果順序選択
16  SPECIAL_CONDITION{specialConditionType}                            特殊状態選択
```

旧 NOTES.md は 7/8/9/13/14 だけ列挙していたが、`cg.api` 公式列挙では 0-16 の 17 種。
特に 10 ABILITY / 11 DISCARD / 12 RETREAT / 15 SKILL / 16 SPECIAL_CONDITION は
中盤以降で頻出するはず。

### SelectType / SelectContext / AreaType / EnergyType / CardType / SpecialConditionType

完全な列挙は `cg/api.py:11-187` を参照。NOTES.md には主要なものだけ抜粋。

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

## 提出フロー

```bash
# 1. tar.gz を作る (main.py + deck.csv + cg/ + train/policy.npz, features.py, policy.py)
./make_submission.sh                     # -> submission.tar.gz

# 2. Kaggle にアップロード
.venv/bin/kaggle competitions submit -c pokemon-tcg-ai-battle \
    -f submission.tar.gz -m "submission note"
```

提出 zip の中身 (検証済み):

```
main.py
deck.csv
cg/{__init__.py,api.py,game.py,sim.py,utils.py,libcg.so,cg.dll}
train/{__init__.py,policy.py,features.py,policy.npz}
```

## 次の打ち手 (優先順)

1. **PIMC (Perfect Information Monte Carlo) を実装** — `cg.api.search_begin/step` で
   情報集合サンプリング。線形ポリシーは rollout policy に流用できる
2. **長め自己対戦** で学習を積む (`--episodes 2000` で勝率カーブを取る)
3. **特徴量追加**: カード ID 埋め込み、active Pokemon の attack 候補
4. **PyTorch + PPO**: 線形ポリシーの限界を超えるとき
5. **デッキ差し替え**: `deck.csv` をメタ環境に合わせて再構成
6. **ストラテジー部門レポート** (締切 9/14)

## 環境メモ

- Python 3.12.8 (devbox の `python3@latest`)
- `kaggle-environments==1.30.1` で `cabt` env 同梱を確認(配布概要には 1.14.10 とあるが、
  `cg/api.py` がサンプル同梱なのでサンプル側を真と見なすこと)
- Kaggle CLI: `.venv/bin/kaggle` (`pip install kaggle` を venv 内、OAuth 認証済み)
- GPU 無し、CPU で numpy 線形ポリシー。500ep ≈ 3分 / 2000ep ≈ 12分
- リモート ephemeral コンテナ前提 (work in progress は push しないと消える)
- 配布 zip は `pokemon-tcg-ai-battle.zip` (~300MB, PDF カードリスト含む)。
  PDF 以外は `kaggle_data/` に展開済み
