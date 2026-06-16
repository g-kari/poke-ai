---
description: Kaggle 提出物の構造要件 (tar.gz / main.py 直下 / deck.csv 同梱) を逸脱すると検証エピソードで死ぬ
paths: "main.py,deck.csv,cg/**,train/policy.npz,make_submission.sh"
---

# 提出物の構造は厳密 — 仕様を確認せずに変えない

## 不変条件

提出は `.tar.gz` ファイル単体。展開時のルート (= ネストされていない top-level) に
必ず以下が存在しなければならない:

- `main.py` — `agent(obs: dict) -> list[int]` を export
- `deck.csv` — 60 行の card ID

Kaggle ランタイムは tar の中身を `/kaggle_simulations/agent/` にマウントする。
これ以外の filename・配置・ディレクトリ階層は通らない。

## なぜ

`main.py` という filename は kaggle-environments の cabt env が `agent()` を
探すために決め打ちしている。`agent.py` では import されない。`deck.csv` は
**初期選択フェーズで `obs["select"] is None` のときに 60 枚を返す**ための
データソースで、コードに card ID をハードコードする運用は競技ルール上 legal だが
バンドル展開時のパス依存 (`/kaggle_simulations/agent/deck.csv`) のため、
実行コードからも参照可能な root 直下に置く約束になっている。

## How to apply

- 新しいエントリ関数を追加するときは `main.py` の中で書く。別ファイルを
  作って `from foo import agent` してはいけない (Kaggle が `main.agent` を解決できない)
- deck を変更するときは `deck.csv` を編集する。`main.py:DECK` のような
  Python リテラルは作らない (同期忘れの事故になる)
- 同梱物を増やすときは `make_submission.sh` の `require` リストと
  `tar -czvf` 引数の両方を更新する。pre-commit hook `check-cg-bundle`
  (`scripts/check_bundle.py`) の `REQUIRED` リストも合わせて更新する
- 検証は `./make_submission.sh /tmp/test.tar.gz && tar -tzf /tmp/test.tar.gz`
  で展開時のルートに `main.py` が出ることを確認する
- `kaggle competitions submit` 前には必ず `pre-commit run --all-files` を通す
