# poke-ai

[PTCGABC / ポケカABC](https://ptcg-abc.pokemon.co.jp/) — [Kaggle: Pokémon TCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle) のシミュレーション部門に提出するエージェント。

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
| **2-MLP ensemble (異 seed, 各 2000ep)** | **37-3 (40 戦, 92.5%)** | vs rule_based +10pp (22.5→32.5%)。default の policy |

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
