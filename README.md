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
| **PPO_v40 seed=100 single (base + 1280ep PPO)** | **23.3% @ 700g (= single PEAK)** | 期待 LB ~815、 明日 UTC で提出予定 |
| PPO_v40 seed=2026 single (= 23% 級 2 個目) | 22.9% @ 700g | s100 と同等 strength、 V6 特化 profile |
| PPO_v40 seed=500 single (= n=4 中 mid-tier) | 18.6% @ 700g | median signal、 ratio 35 検証用 |
| PPO_v40 seed=42 single (= n=4 中最弱) | 16.3% @ 700g | train 累積 26.9% (最高) なのに bench 最低 = overfit signal |

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
| 4-MLP ensemble (seed=0/2/100/200) | **558.2** | DL (still settling +20.2/h、 目標 752) |
| Mixed (seed=0 ext + 2/100 base) | 470.9 | DL |
| Mix v3 (seed=100 ext + 0/2 base) | 411.8 | DL |
| Alt v3 (seed=2/100/300、 no seed=0) | 184.0 | DL (=構造的失敗) |

**🚨 訂正**: 前回スナップショットで Mixed が LB 711.2 と観測しましたが、
TrueSkill σ settling 後 **490.3 まで下降**。 transient peak でした。
3-MLP base 679.6 がチャンピオン継続です。

**LB ↔ lab ratio (= 7-opp suite × 700g 大規模 bench で校正)**:

| 提出 | 700g lab | LB | ratio |
|---|---|---|---|
| **3-MLP base** | **19.3%** | **679.6** | **35.2** |
| BCRL2 | **16.1%** | 570.4 | **35.4** |
| V60 EXT3 | **17.0%** | 562.4 | **33.1** |

**🎯 重要発見**: 大規模 bench 3 サンプルで ratio が **33-35 の narrow range
に収束**。 「3-MLP base が特別な特殊解 ratio 35.9」 とは思っていたが、
実は **DL submission の lab → LB は ratio ~35 で線形対応** が正しい
解釈。 旧 280g sample で見えた ratio 22-29 の散らばりは **bench noise +
TrueSkill σ settling** の混入だった。

**Practical implication**: lab 20% (= 700g 真値) を達成すれば LB ~700。
PPO で BCRL2 (16.1%) → 18-20% への改善が可能なら LB 630-700 が現実
的視野。

### 明日の path (UTC reset 後の 5 slot、 4 枠投入計画)

1. **PPO_v40 seed=100 single 提出** (= lab 23.3% PEAK、 Crustle Wall 特化
   profile、 期待 LB ~815、 ratio 35 仮説の検証 1)
2. **PPO_v40 seed=2026 single 提出** (= lab 22.9% 同 strength、 V6 特化
   profile、 期待 LB ~800、 ratio 35 仮説の検証 2 + specialization 効果)
3. **PPO_v40 seed=500 single 提出** (= lab 18.6% mid-tier、 期待 LB ~650、
   "lab → LB 線形性" の中間点校正)
4. **3-MLP base 再提出 (control)** — settling 後の真 LB 確認 + ratio 35
   の baseline (= 既知 LB 679.6)
5. **残り 1 slot adaptive** — 上記 1-4 の LB 結果次第で決定

**期待される知見** (LB 着地点から判明する事):
- s100 ≈ s2026 ≈ 815 → ratio 35 確定、 PPO 高 lab が高 LB に直接 translate
- s100 vs s2026 で大差 → matchup specialization が LB に強く効く →
  meta-deck 分布の理解が次の鍵
- 全部 LB < 700 → "PPO 23% lab は overfit 仮説"、 ratio 25-30 へ下方修正
- s500 が ratio 35 → "lab → LB 線形" 確定

**今日確定した重要な学び**:
- PPO は ~50% の確率で 23% 級を引く ガチャ性質 (n=4 試行で確認)
- ensemble は v40 PPO 系統では原理的に機能しない (4 失敗パターン確定):
  - features 起因 (v60 deck hash)
  - PPO 収束で seed diversity 消失
  - policy strength 不均衡で dilution
  - specialization 境界で confusion (= 一部 matchup で両 single より下)
- → **single policy submission が確定的に最適戦略**

**驚きの発見 (2026-06-19)**: 5 個の DL submission 試行 (4-MLP / Mixed /
Mix v3 / Alt v3) すべて ratio 22-23 で 3-MLP base の 35.9 を再現できず:

| 試行 | 構成 | lab | LB | ratio |
|---|---|---|---|---|
| **3-MLP base** | 0/2/100 | 18.9% | **679.6** | **35.9** |
| 4-MLP base | 0/2/100/200 | 20.4% | 502.3 | 22.2 |
| Mixed | **0 ext**/2/100 | 20.4% | 470.9 | 23.1 |
| Mix v3 | 0/2/**100 ext** | 19.4% | 411.8 | 21.2 |
| Alt v3 | 2/100/**300** (no 0) | 22.2% | 515.8 | 23.2 |

**最終結論 (重要な構造的発見)**:
- 3-MLP base (seed=0/2/100 base only) は **特定の seed 組合せで生じる
  特殊解**。 ratio 35.9 = 我々の最高効率
- 「より多くの seed」「seed の差し替え」「entropy 追加」 のいずれも
  ratio を **23 付近に着地** させ、 LB 500 前後に強制収束
- ensemble の LB 効率は **seed 多様性の偶然のマッチ** で決まる、 一意
  に再現できない

**注**: 3-MLP base の ratio 35.9 は群を抜く。 ensemble の diversity
が LB の多様な相手分布に強い証拠。 Mixed (1 seed だけ entropy ext) は
lab 改善 (+1.5pp) したが ratio 低下 (-3.6)、 結果 LB -19.9pp の trade。

### 重要な構造的発見 (2026-06-19/20、 補遺 14 + PIMC Phase 1 完了時点)

1. **3-MLP の lab は bench 依存**: bench_meta.py (4 opp) で 26.7%、
   bench_v40.py (7 opp) で **18.9%**。 LB との対応は 7-opp の方が良い
2. **PPO ガチャ性質 ~50% で 23% 級**: n=4 試行 (seed 0/42/500/2026)
   で median 20.75%、 mean 20.3%、 std 3.5pp。 seed=100 と seed=2026 が
   両方 23% 級 (= ~50% 確率で当たる ガチャ)
3. **全 PPO_v40 PEAK は 2nd stage で必ず劣化** (s100 ext -3.4pp、
   s2026 ext -3.5pp): lab PEAK な policy は触らない方が良い
4. **v40 PPO ensemble は何をやっても機能しない (5 失敗パターン)**:
   - (a) features 起因 (v60 deck hash)
   - (b) PPO 収束で seed diversity 消失
   - (c) policy strength 不均衡 dilution
   - (d) specialization 境界 confusion (同 mode = 補遺 6)
   - (e) mode mismatch confusion (異 mode = 補遺 13)
5. **PPO 学習は 2 軸 trade-off** (= 補遺 10-14 で完全証拠):
   - 軸 A: rule-based specialization (= lab metric)
   - 軸 B: 中庸 player (= 3-MLP base) への対応力 (= 1v1 winrate)
   - lab と 1v1 winrate に **負の相関**: s100 (23.3% → 37.5%), s2026
     (22.9% → 40.0%), s500 (18.6% → **77.5%**), s42 (16.3% → 55.0%)
   - **3-MLP base ですら s500 に 1v1 で 22.5% で大敗** = DL champion
     の優位は rule-based pool 専用
6. **非推移性 (じゃんけん的)**: s500 > 3-MLP base, 3-MLP base > s100/s2026,
   s100 ≈ s500 ≈ s2026 — strategic depth は単一軸ではない
7. **Single policy submission が確定的に最適戦略**: 明日 UTC 4 枠投入
   (s100/s2026/s500/3-MLP base) で 2 軸 model を LB 上で検証
8. **PIMC Phase 1 完了**: cg.api.search_begin/step/release 動作確認 OK
   (scripts/pimc_smoke_test.py PASSED)。 LB 着地後に Phase 2 着手予定

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
