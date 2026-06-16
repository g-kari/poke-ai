---
description: コミット前の品質ゲート — pre-commit 必須、--no-verify 禁止、policy.npz は明示 commit
paths: ".pre-commit-config.yaml,scripts/check_*.py,train/policy.npz"
---

# コミット品質ゲートを迂回しない

## 不変条件

- `pre-commit install` 済の状態で開発する
- `git commit --no-verify` / `git commit -n` は使わない
- 提出に必要な `train/policy.npz` は明示的にコミットする (`.gitignore` の
  `*.npz` の exception になっている)
- `train/metrics_*.json` も同様にコミット (学習履歴を git log で追えるように)

## なぜ

pre-commit hook が以下を保証している:

| hook | 検知すること |
|---|---|
| `validate-deck-csv` | deck.csv が 60 行か |
| `check-main-py` | main.py が import 可能で agent() が 60 枚返すか |
| `check-cg-bundle` | 提出に必要な全ファイルが揃っているか |
| `ruff` (`--fix`) | Python lint + 自動修正 |
| `ruff-format` | フォーマット統一 |
| `trailing-whitespace` / `end-of-file-fixer` | 行末・EOF 整形 |
| `check-added-large-files` | 5MB 超のファイルを止める |

`--no-verify` で迂回すると検証エピソードで死ぬ提出を push してしまう。
deck.csv が 59 行 / 61 行になっていた、main.py が import errors を出す、
cg/ から ファイルが消えている、といった事故が hook で全部止められる。

## How to apply

- どうしても hook を skip したいときは個別の `SKIP=hook-id git commit ...`
  で **一時的に対象を絞る** こと。全体を `--no-verify` で迂回しない
- 緊急時の例外を作るなら commit message に理由を明記:
  `commit: ... (SKIP=check-main-py; reason: WIP refactor mid-way)`
- ruff が touch するのは `cg/`, `kaggle_data/`, `.venv/` を除外した範囲のみ。
  ruff-format で勝手に書き換わったときは内容を確認 (副作用がないか) して
  そのまま commit に取り込む
- `train/policy.npz` を更新したら `git diff --stat` で size 変化と
  `train/metrics_*.json` の有無を確認してからコミット。policy.npz だけ
  上げて metrics 忘れる事故が起きやすい
