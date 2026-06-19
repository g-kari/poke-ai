# CLAUDE.md

開発時に Claude Code が読む補助メモ。プロジェクト固有の事情と"知っておかないと事故る"罠を集約。

## TL;DR (30秒で把握する)

- **何か?** Kaggle ポケモン TCG AI バトル (シミュ部門) の提出エージェント
- **提出は?** `./make_submission_*.sh` → `*.tar.gz` (1.0-1.2MB) → `kaggle competitions submit`
- **エントリ?** リポジトリ root の `main.py` (filename 固定、ネスト不可)
- **学習は?** `scripts/run.sh python3 -m train.ppo_train_v40 ...` (PPO + v40 features、 GPU、 ~9分/1280ep)
- **強さ (DL)?** LB 679.6 = 3-MLP base ensemble (= champion)、 PPO_v40 seed=100 lab 23.3% = single PEAK
- **強さ (rule-based)?** LB 874.7 = CrustleDashimaki (= 全体 champion)
- **PIMC?** `cg/api.py` の `search_begin/step/release` で実装可能 (凍結解除済)、 未着手
- **GPU?** RTX 3070 Ti (8GB) + PyTorch cu128 動作確認済。`scripts/env.sh` が `libcuda` を解決
- **コミット前?** `pre-commit install` 必須 (deck/main/bundle/ruff の品質ゲート)
- **次の打ち手?** §"次の打ち手" を見るべし

## 運用ルール (`.claude/rules/`)

具体的な「やる/やらない」のルールは `.claude/rules/*.md` に分離している。
作業前に該当ファイルを開いて読むこと。

| ファイル | いつ読む |
|---|---|
| [`submission-format.md`](.claude/rules/submission-format.md) | `main.py` / `deck.csv` / `make_submission.sh` を編集するとき |
| [`deck-rules.md`](.claude/rules/deck-rules.md) | `deck.csv` を変更するとき |
| [`python-env.md`](.claude/rules/python-env.md) | Python を起動するとき (素の `python3` を打つ前に) |
| [`vendored-cg.md`](.claude/rules/vendored-cg.md) | `cg/` 配下に触れるとき (= 触らない方針の確認) |
| [`engine-quirks.md`](.claude/rules/engine-quirks.md) | エージェント関数 / 学習ループを書くとき |
| [`cg-api-priorities.md`](.claude/rules/cg-api-priorities.md) | ABI を調べるとき (pip 版・docs・cg/ の優先順) |
| [`commit-gate.md`](.claude/rules/commit-gate.md) | `git commit` する前 |
| [`rule-maintenance.md`](.claude/rules/rule-maintenance.md) | ルール自体を編集するとき |

## サブエージェント (`.claude/agents/`)

繰り返し性が高く、責務が明確に切れる作業は専用エージェントに委譲する。

| エージェント | いつ使う |
|---|---|
| [`trainer`](.claude/agents/trainer.md) | 学習を回す + A/B テストで baseline と比較 + 悪化したら復元 |
| [`bencher`](.claude/agents/bencher.md) | 勝率測定 (vs random, vs 別 policy) + 95% 信頼区間 |
| [`submission-validator`](.claude/agents/submission-validator.md) | `kaggle submit` 前の tar.gz 構造 + 動作検証 |
| [`engine-explorer`](.claude/agents/engine-explorer.md) | `cg/api.py` / カードデータの factual な質問 (実装は禁止) |

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

### 環境セットアップ (初回のみ)

nix の Python ランタイムは numpy の C 拡張ロード時に libstdc++ / libz が見つから
ない問題があるため、`scripts/env.sh` で nix store から動的に解決してから python を
起動する。`scripts/run.sh <cmd>` がワンライナのラッパー。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install kaggle kaggle-environments numpy

# pre-commit 品質ゲート (deck 60 行 / main.py import / bundle 完全性 / ruff)
pre-commit install
```

### 日常コマンド

```bash
# スモークテスト (~1秒 / 8試合)
scripts/run.sh python3 selfplay_test.py 4

# 学習 (~10分 / 2000ep)、policy.npz から warm-start
scripts/run.sh python3 -m train.reinforce \
    --episodes 2000 --lr 0.05 \
    --warm-start train/policy.npz \
    --out train/policy.npz \
    --metrics-out train/metrics_2000ep.json

# 学習済み重みは train/policy.npz に保存され、main.py が起動時に自動ロード
```

## ファイル構成

```
main.py               # Kaggle 提出エントリ。agent(obs)→list[int]。policy.npz があれば使う / 無ければ engine 順
agent.py              # ローカル開発用の旧エントリ (selfplay_test 等が参照)。提出には使われない
deck.csv              # 提出デッキ 60 行 (card ID per line)
pyproject.toml        # ruff 設定 (line=100, py312, litellm shim 用 E402 例外)
.pre-commit-config.yaml  # コミット時の品質ゲート設定
scripts/
  env.sh              # source 用: .venv 有効化 + nix libstdc++/libz を LD_LIBRARY_PATH に
  run.sh              # ラッパー: scripts/run.sh <cmd>
  check_deck.py       # deck.csv が 60 行か検証 (pre-commit hook)
  check_main.py       # main.py が import 可能か + agent() が 60 枚返すか (pre-commit hook)
  check_bundle.py     # 提出に必要なファイルが揃っているか (pre-commit hook)
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
- `pre-commit install` 必須。品質ゲートが通らないコミットは push しない。
  `--no-verify` で迂回する場合は commit message に理由を明記すること。
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

## 次の打ち手 (優先順、 2026-06-19 更新)

1. **明日 UTC reset 後に PPO_v40 seed=100 single 提出**
   (`submission_ppo_v40_s100.tar.gz` 検証済、 lab 23.3% PEAK、 期待 LB ~815)
   - 結果次第で ratio 35 仮説の校正点 = LB 815 確認なら成功
   - LB < 700 着地なら "seed=100 overfit 仮説" が成立
2. **PPO 探索の median-of-N protocol 導入**: 同じ手法で 3-5 seed 試行し
   median lab を真の signal とする (= seed=500 で 18.6% 出た教訓)
3. **PIMC (Perfect Information Monte Carlo)**: `cg.api.search_begin/step` で
   情報集合サンプリング。 PPO_v40 seed=100 を rollout policy に流用
4. **AlphaZero (PIMC + value head)**: PPO_v40 の value network を learnt
   value として PIMC に組込み、 search のリーフ評価を改善
5. **デッキ差し替え**: `deck.csv` をメタ環境に合わせて再構成 (= deck と
   policy の strong coupling を活かす)
6. **ストラテジー部門レポート** (締切 9/14、 STRATEGY_REPORT.md は v2 完成済、
   ratio 35 検証結果待ち)

## 終わった打ち手 (= 試行済み、 結論あり)

- ✅ PPO Phase 1-5 完成 (= LB 700 突破の最有力路線として実装) — PPO1 LB
  570.4 で完了、 v60 chain 10 variants 試行
- ✅ PPO_v40 chain (seed=0/2/100/500): single PEAK 23.3% (seed=100)、
  ensemble 失敗 3 パターン (PPO 収束 / strength 不均衡 / random seed dependency)
- ✅ 3-MLP ensemble (= LB 679.6, ratio 35.9) — 我々の DL champion
- ✅ V60 features (= 60-d deck fingerprint) — 7 サイクル投資、 ensemble
  効果なし、 BCRL2 LB 570.4 が best
- ✅ BC + REINFORCE (= AlphaGo 縮小版) — LB 570.4 で打ち止め
- ✅ Deck builder GA + PIMC v1-v5 + Crustle 検出 — heuristic value の天井

## 環境メモ

- Python 3.12.8 (devbox の `python3@latest`)
- `kaggle-environments==1.30.1` で `cabt` env 同梱を確認(配布概要には 1.14.10 とあるが、
  `cg/api.py` がサンプル同梱なのでサンプル側を真と見なすこと)
- Kaggle CLI: `.venv/bin/kaggle` (`pip install kaggle` を venv 内、OAuth 認証済み)
- **GPU**: RTX 3070 Ti (8 GB), CUDA 13.1 driver via WSL2。
  `scripts/env.sh` が `/usr/lib/wsl/lib` を `LD_LIBRARY_PATH` に追加するので、
  `scripts/run.sh python3 -c 'import torch; print(torch.cuda.is_available())'` → True
- **PyTorch**: `2.11.0+cu128` (CUDA 12.8 wheel) を venv にインストール済み。
  numpy 線形ポリシーの限界が来たら `train/` を PyTorch (MLP / PPO / value head) に置き換えること。
- 線形ポリシー学習速度: CPU で 2000ep ≈ 6 分 (warm-start 込み)
- リモート ephemeral コンテナ前提 (work in progress は push しないと消える)
- 配布 zip は `pokemon-tcg-ai-battle.zip` (~300MB, PDF カードリスト含む)。
  PDF 以外は `kaggle_data/` に展開済み
