---
description: deck.csv のフォーマット / 枚数 / カード制約 — pre-commit が落ちる前に守る
paths: "deck.csv,scripts/check_deck.py"
---

# deck.csv は競技ルールに従う厳密なフォーマット

## 不変条件

- 行数: **ぴったり 60**。trailing 空行も含めて 60 を超えてはならない
- 各行: card ID の正の整数 (`int(line.strip()) > 0` が真)
- 同じ card ID は **基本エネルギー以外** で最大 4 枚まで
- ACE SPEC カードはデッキ全体で最大 1 枚 (`Card Data CSV` の `Rule` 列で識別)

## なぜ

公式ルールの 60 枚 / 4 枚制限を破ったデッキは検証エピソードで
"Invalid deck" として弾かれ、提出が `Errored` 扱いになる。
基本エネルギー (Basic Energy) のみ枚数制限なし — `EN_Card_Data.csv` の
`Stage (Pokémon)/Type (Energy and Trainer)` 列が `Basic Energy` のものが該当。

## How to apply

- card ID 重複数を確認: `awk '!/^$/{c[$0]++} END{for(k in c)if(c[k]>4)print k,c[k]}' deck.csv`
- 基本エネルギーかどうかは `kaggle_data/EN_Card_Data.csv` で
  `Stage` 列を grep する: `awk -F, '$5=="Basic Energy"' kaggle_data/EN_Card_Data.csv`
- pre-commit hook `validate-deck-csv` (`scripts/check_deck.py`) は行数と
  正数チェックだけ。4 枚制限・ACE SPEC 制限は手動で確認すること
- デッキ変更時はベンチマーク `scripts/run.sh python3 selfplay_test.py 20`
  を必ず通して勝率退行を見る (新デッキで学習済み policy が機能しない可能性)
- deck.csv を変更したらほぼ確実に学習し直し。新しい card ID 分布は features.py の
  hash buckets と相互作用する
