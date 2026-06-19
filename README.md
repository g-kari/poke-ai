# poke-ai

[PTCGABC / ポケカABC](https://ptcg-abc.pokemon.co.jp/) — [Kaggle: Pokémon TCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle) のシミュレーション部門に提出するエージェント。

📊 **メタデッキ・ダッシュボード**: <https://g-kari.github.io/poke-ai/>
(対戦相性表、LB 環境、提出履歴を視覚化。`docs/` 配下を GitHub Pages で配信)

📝 **ストラテジー部門レポート (締切 2026-09-14)**: [STRATEGY_REPORT.md](STRATEGY_REPORT.md)
30 サイクル × 30 分 ≈ 15 時間の探索ログを 12 系統 × 5 deep-dive + 3 中心的学び で整理。

📐 **PPO 実装計画 (未着手)**: [docs/PPO_DESIGN.md](docs/PPO_DESIGN.md)
V60 振動制御の本命路線、 後継者が引き継げる設計書。

🔄 **定期更新**: `scripts/cron_update_pages.sh` を呼ぶと kaggle CLI から最新 LB を
取得して docs/index.html を更新 + git push (差分なしならスキップ)。 cron / /loop で
定期実行に組み込めます。 `scripts/update_pages.py --no-fetch` でオフライン更新も可。

`obs` (`dict`) を受け取り option index のリストを返す `main.py` をエントリに、
self-play REINFORCE で学習した numpy 線形ポリシー (`train/policy.npz`) を内蔵。

## 締切

| 日付 (UTC 23:59) | 内容 |
|---|---|
| 2026-08-09 | 応募 / チーム合併 |
| **2026-08-16** | **最終提出** |
| 2026-08-17 〜 08-31 | ランキング確定 |
| 2026-09-14 | ストラテジー部門レポート |

毎日 5 件まで提出可、最新 2 件のみ追跡。初期スコア μ₀ = 600 の TrueSkill 風。

## クイックスタート

```bash
# 仮想環境を作る (初回のみ)
python3 -m venv .venv
source .venv/bin/activate
pip install kaggle kaggle-environments numpy
pre-commit install   # コミット時の品質ゲート

# scripts/run.sh は nix の Python + WSL2 の libcuda パスを補完するラッパー
scripts/run.sh python3 selfplay_test.py 4
scripts/run.sh python3 -m train.reinforce \
    --episodes 2000 --lr 0.05 \
    --warm-start train/policy.npz \
    --out train/policy.npz \
    --metrics-out train/metrics_2000ep.json
```

学習済み重みは `train/policy.npz`、`main.py` 起動時に自動ロードされる。

## 提出

```bash
./make_submission.sh                                    # → submission.tar.gz (1.1MB)
.venv/bin/kaggle competitions submit \
    -c pokemon-tcg-ai-battle \
    -f submission.tar.gz \
    -m "submission note"
```

`submission.tar.gz` の中身 (zip ルート直下に `main.py` 必須):

```
main.py            agent(obs) → list[int]
deck.csv           60 行の card ID
cg/                公式同梱の Python 実装 + libcg.so + cg.dll
train/             policy.npz + 推論用の policy.py / features.py
```

## 構成

```
main.py               Kaggle 提出エントリ (filename 固定)
agent.py              ローカル開発の旧エントリ (selfplay_test 等が import)
deck.csv              提出デッキ 60 行
cg/                   公式 sample_submission/cg/ をそのまま vendor
selfplay_test.py      vs random ベンチ
make_submission.sh    .tar.gz バンドル生成
train/
  features.py         state 36-d / option 36-d 特徴
  policy.py           numpy 線形ポリシー
  reinforce.py        self-play REINFORCE
  policy.npz          学習済み重み (コミット対象)
  metrics_*.json      学習履歴
scripts/
  env.sh              source 用: 仮想環境 + nix lib + WSL libcuda の LD_LIBRARY_PATH 設定
  run.sh              ラッパー: scripts/run.sh <cmd>
  check_*.py          pre-commit 品質ゲート用
.pre-commit-config.yaml
pyproject.toml        ruff 設定
NOTES.md              実機調査ログ (obs スキーマ, OptionType 列挙, search_begin の API)
CLAUDE.md             Claude Code 向け開発メモ
```

## 現状の強さ

| バージョン | vs random | 備考 |
|---|---|---|
| engine 順フォールバック | 14-2 (16 戦) | option index 0 を常に選ぶだけ |
| 500ep 学習 (state 24-d / opt 18-d) | 19-5 (24 戦) | PR #1 |
| 500ep + Pokemon-aware 特徴 (36-d / 36-d) | 23-1 (24 戦) | PR #2 |
| 2000ep warm-start | 34-6 (40 戦, 85%) | A/B で 500ep の 31-9 を上回り |
| 5000ep cumulative | 36-4 (40 戦, 90%) | 2000ep からの warm-start |
| 2000ep + ATTACK dmg/cost 特徴 (36-d / 40-d) | 65-15 (80 戦, 81%) | 5000ep を 2.5x 少ない episode で同等 |
| 5000ep cumulative + ATTACK 特徴 | 70-10 (80 戦, 87.5%) | 新特徴量で warm-start 続行 |
| 5000ep + super-effective / retreat 特徴 (40-d / 40-d) | 72-8 (80 戦, 90.0%) | 線形 100-game 91% で飽和 |
| PyTorch MLP (64→32) + 2000ep self-play | 38-2 (40 戦, 95%) | 線形に mirror match で 23-17 (57.5%) 勝利 |
| 2-MLP ensemble (異 seed, 各 2000ep) | 37-3 (40 戦, 92.5%) | vs rule_based +10pp (22.5→32.5%) |
| **3-MLP ensemble (seed=20260628 + 42 + 100, 各 2000ep)** | **vs 4 meta deck 32-88 (120 戦, 26.7%)** | 2-MLP の 22.5% から +4.2pp。default の policy |
| V60 EXT3 (10500ep, features_v60 60-d) | 20.5% @ 30g/opp | LB 573.9 — 3-MLP (679.6) を下回り |

### LB 履歴 (最新スナップショット 2026-06-19、 **訂正版 — TrueSkill settling 後**)

| Submission | スコア | 種別 |
|---|---|---|
| CrustleDashimaki | **874.7** | 🥇 rule-based ベスト |
| V6 (Crustle+Lucario hybrid) | 860.8 | rule-based |
| Iono | 762.2 | rule-based |
| 3-MLP ensemble (seed=0/2/100 base) | **679.6** | 🥇 DL ベスト (継続) |
| 2-MLP ensemble | 613.3 | DL |
| BCRL2 (BC v2 + REINFORCE 7000ep) | 570.4 | DL |
| V60 EXT3 (single 10500ep) | 562.4 | DL |
| Mixed (seed=0 ext + 2/100 base) | 490.3 | DL (失敗) |
| Mix v3 (seed=100 ext + 0/2 base) | 404.5 | DL (失敗) |

**🚨 訂正**: 前回スナップショットで Mixed が LB 711.2 と観測しましたが、
TrueSkill σ settling 後 **490.3 まで下降**。 transient peak でした。
3-MLP base 679.6 がチャンピオン継続です。

**LB ↔ lab ratio (= 7-opp suite、 settling 後の確定値)**:

| 提出 | 7-opp lab | LB | ratio |
|---|---|---|---|
| **3-MLP base** | **18.9%** | **679.6** | **35.9** (最高効率 = チャンピオン) |
| BCRL2 | 19.3% | 570.4 | 29.5 |
| V60 EXT3 | 20.5% | 562.4 | 27.4 |
| Mixed (Mix v1) | 20.4% | 490.3 | 24.0 (ext seed が ensemble 破壊) |
| Mix v3 | 19.4% | 404.5 | 20.9 |

**結論**: lab winrate が高くても ensemble に ext seed を混ぜると LB
ratio が大幅低下。 entropy + warm-start で deterministic になった seed
は ensemble の中立性 (= 多数決ではなく logit 平均) を破壊する。

**真理**: 3-MLP base (純粋 exploratory ensemble) が最適。 ext は
**single policy 提出** に有効、 ensemble には不適。

**注**: 3-MLP base の ratio 35.9 は群を抜く。 ensemble の diversity
が LB の多様な相手分布に強い証拠。 Mixed (1 seed だけ entropy ext) は
lab 改善 (+1.5pp) したが ratio 低下 (-3.6)、 結果 LB -19.9pp の trade。

### 重要な構造的発見 (2026-06-19)

1. **3-MLP の lab は bench 依存**: bench_meta.py (4 opp) で 26.7%、
   bench_v40.py (7 opp) で **18.9%**。 LB との対応は 7-opp の方が良い
2. **個別 seed 改善 ≠ ensemble 改善**: entropy_coef=0.02 で個別 v40
   seed は lab +3.6~+7.3pp 改善 (= 単独 ratio 期待)、 だが 3 個全部 ext
   した ensemble は lab 16.1% で **-10.6pp regression**
3. **ensemble は exploratory policies に強い**: entropy 抑えた base
   policies は logit averaging で良い中庸を取れる、 entropy + ext は
   確信的になり averaging が破綻
4. **「1 ext + 2 base」の mixed が良い妥協点**: lab 20.4%, LB 659.7

## 環境

- **OS**: WSL2 (Ubuntu) on Windows
- **GPU**: NVIDIA RTX 3070 Ti (8GB), CUDA 13.1 driver
- **Python**: 3.12.8 (devbox `python3@latest`)
- **PyTorch**: `2.11.0+cu128` (CUDA 動作確認済)。線形ポリシーの限界が来たら MLP / PPO に切り替え可能
- **Kaggle CLI**: `.venv/bin/kaggle` (`~/.kaggle/credentials.json` で OAuth 済)

## ドキュメント

- 公式 API: <https://matsuoinstitute.github.io/cabt/>
- 実機調査の詳細 (obs スキーマ全体、OptionType 全列挙、IS-MCTS のための `search_begin` 使い方) → [`NOTES.md`](./NOTES.md)
- 開発フロー / 罠 / コミット規約 → [`CLAUDE.md`](./CLAUDE.md)

## 次の打ち手

1. **PIMC / IS-MCTS** — `cg.api.search_begin / search_step / search_release` で情報集合サンプリング。線形ポリシーは rollout policy に流用
2. **長め学習** (5000ep, lr scheduler 付き)
3. **PyTorch MLP** policy (GPU 活用)
4. **特徴量強化** (`all_card_data()` でカード ID 埋め込み)
5. **デッキ最適化** (`kaggle_data/EN_Card_Data.csv` からメタ環境構築)
6. **ストラテジー部門レポート** (締切 2026-09-14)

## ライセンス

Pokémon / Nintendo / Creatures / GAME FREAK ほか各社の商標。
