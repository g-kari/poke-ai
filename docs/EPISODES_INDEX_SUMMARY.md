# 公式 Top Episodes Dataset 探索メモ

## 発見 (2026-06-22)

Kaggle 公式が日別 dataset として LB トップ層の対戦リプレイを公開
(= STRATEGY_REPORT 補遺 25 参照)。 利用方針:

- `pokemon-tcg-ai-battle-episodes-index`: 日付別の manifest (= 483 B、 軽量)
- `pokemon-tcg-ai-battle-episodes-YYYY-MM-DD`: 1 日分 (= 数 GB)

## Manifest 解析 (2026-06-22 取得)

| date | episode_count | total_bytes | top_avg_score | median_avg_score |
|------|---------------|-------------|---------------|------------------|
| 2026-06-16 | 1,277  |  2.85 GB | 1024.6 | 627.8 |
| 2026-06-17 | 7,819  | 21.47 GB | 1259.8 | 761.0 |
| 2026-06-18 | 6,532  | 21.47 GB | 1327.1 | 926.6 |
| 2026-06-19 | 5,426  | 21.47 GB | 1324.9 | 1013.7 |
| 2026-06-20 | 5,178  | 21.47 GB | 1311.2 | 1063.8 |
| 2026-06-21 | 5,054  | 21.47 GB | 1332.0 | 1110.7 |

**観察**:
- 6/16 が一番軽い (2.85 GB) = 探索の最初の足場として最適
- top_avg_score (= LB トップ層の平均) は 1024 → 1331 で日々上昇
  (= コンペが熟成するほど強い player が混ざる)
- median_avg_score も 627 → 1110 と上昇 (= 全体的に強くなっている)
- 6/16 でも既に top_avg_score 1024 = LB 700+ を超えるトップ層を含む

## 取得方針

1. **第 1 段階** = 6/16 dataset DL (~2.85 GB)
2. **第 2 段階** = `episodes.csv` 抽出、 LB top 10% を絞り込み
3. **第 3 段階** = `strong_team_visible_cards.csv` で deck 採用率推定 →
   現状の `deck.csv` と比較
4. **第 4 段階** = BC v2 教師データ生成 (= action_sample.csv の winner
   side のみ抽出して学習)

## 注意点

- `kaggle_data/` は `.gitignore` で除外済 → リポジトリには含まれない
- 個別 episode JSON は `episode-*.json` glob で除外
- DL は `.venv/bin/kaggle datasets download` (OAuth 認証済)
- 1 episode = ~2 MB 平均、 6/16 で 1277 episodes
