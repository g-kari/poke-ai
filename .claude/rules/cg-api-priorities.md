---
description: ABI / 仕様の真実は cg/api.py — pip の kaggle_environments と公式 docs の優先順
paths: "cg/**,main.py,train/**"
---

# ABI の真実は cg/api.py、それ以外は参考

## 優先順 (高い順)

1. **`cg/api.py`** (= `sample_submission/cg/api.py` の vendored copy)
2. **公式 docs** <https://matsuoinstitute.github.io/cabt/>
3. `pip install kaggle-environments` の `kaggle_environments.envs.cabt.cg.*`
4. HANDOVER.md / 旧 NOTES.md / 旧 CLAUDE.md の記述

## なぜ

- `cg/api.py` は **Kaggle ランタイムで実際に使われる** コードそのもの
  (提出時に tar に同梱する)。`pip install kaggle-environments` 同梱の
  `kaggle_environments.envs.cabt.cg.*` とは中身が一致しない (バージョン差)。
  提出環境で動くコードを書くには `cg/api.py` の方を真として読む
- 公式 docs (上記 URL) は人間向けで、`OptionType (0-16)` 等の enum 値や
  `search_begin` のシグネチャを把握するのに最適だが、コードと食い違ったら
  コードが正
- pip 配布版は `cabt.api` モジュールが存在しない (古い HANDOVER の前提)
  ので、ABI 確認用に頼ってはいけない。`battle_start/select/finish` の
  3 関数だけは pip 版でも動く (ローカル self-play で使っているのはこれ)
- 旧ドキュメントの「ABI 未公開」「PIMC は凍結」は誤情報。`cg/api.py` で
  `search_begin/step/release` 全部公開済

## How to apply

- 新しい dataclass や enum 値を参照するときは `cg/api.py` を Read で開いて
  実物を見る (`OptionType` の `:120`, `SearchState` の `:448`, …)
- 「これって使えるんだっけ?」と疑問が出たら、まず
  `grep -nE "^(class |def )" cg/api.py` で実装の全体マップを取る
- pip 配布版を import するのはローカル self-play のとき (engine の
  `battle_start/select/finish` を `kaggle_environments.envs.cabt.cg.game`
  から呼ぶ場合) だけ。実際の ABI 確認や提出用コードは `from cg.api import ...`
- HANDOVER 由来の記述に出くわしたら疑う。「`cabt.api` モジュール」「ABI 未公開」
  「SearchBegin が Python から呼べない」は全て古い情報
- 公式 docs を fetch するときは <https://matsuoinstitute.github.io/cabt/api.html>
  (個別モジュールページ) が enum 一覧として読みやすい
