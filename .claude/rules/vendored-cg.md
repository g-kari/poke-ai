---
description: cg/ は公式 sample_submission の vendored copy — 編集すると upstream 追従不能になる
paths: "cg/**"
---

# cg/ は upstream の vendor で、編集しない

## 不変条件

`cg/` 配下の以下のファイルは公式 `sample_submission/cg/` (Kaggle 配布) の
バイト単位のコピー:

- `__init__.py` (空)
- `api.py` (Observation 等の dataclass + 全 wrapper)
- `game.py` (`battle_start / select / finish / visualize_data`)
- `sim.py` (`ctypes` 経由の `libcg.so` バインディング)
- `utils.py` (JSON ↔ dataclass 変換)
- `libcg.so` (Linux 用 C 本体)
- `cg.dll` (Windows 用 C 本体)

これらは編集してはいけない。コードを改善したいなら別ファイルに wrapper を
書くこと (例: `train/cg_helpers.py`)。

## なぜ

upstream (Kaggle 配布の `sample_submission/cg/`) が将来更新されたとき、
ローカル編集があると diff が取れず、merge が困難になる。Kaggle ランタイム
側で挙動が変わってもこちらで気づけない。`cg/api.py` の dataclass や enum 値が
upstream 側で書き換わったときに、ローカル fork した分は壊れる。

また `libcg.so` / `cg.dll` はバイナリで、こちらでビルドできない (ABI 非公開)。
これを差し替える手段はないので vendor を信用するしかない。

## How to apply

- `cg/` を変更したくなったら、まず「これは upstream に上げるべき修正か?」を考える。
  もし yes なら issue として記録、no なら `train/` 配下に helper を書く
- 提出物作成時の `make_submission.sh` は `cg/` をそのまま tar に入れる。
  ローカル編集してしまった場合は `cp kaggle_data/sample_submission/cg/*.py cg/`
  で原本を復元する
- pre-commit の `trailing-whitespace` / `end-of-file-fixer` / `ruff` は
  `cg/` を exclude 設定済 (`.pre-commit-config.yaml` および
  `pyproject.toml` の `[tool.ruff] exclude`)。新しい hook を追加するときも
  `cg/` を exclude すること
- `cg.api` を import するコードを書くときは「これは local の cg/ から
  読まれる」「Kaggle 提出時も同じ cg/ が tar に入る」と意識する。
  ローカル `kaggle_environments.envs.cabt.cg` とは別物 — そちらは
  Kaggle ランタイムでは利用可能性が保証されない
