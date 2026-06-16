---
description: Python 起動は scripts/run.sh 経由 — nix の Python は素の python3 で numpy/torch が壊れる
paths: "*.py,scripts/**,**/*.sh"
---

# Python 実行は scripts/run.sh ラッパー必須

## 不変条件

ローカルで Python を起動するときは `scripts/run.sh python3 ...` を使う。
直接 `python3 ...` を呼ぶと numpy の C 拡張ロード時に
`ImportError: libstdc++.so.6` または `libz.so.1` で死ぬ。GPU 利用時は
さらに `libcuda.so.1: cannot open shared object file` も追加で発生する。

## なぜ

devbox / nix-built の Python は dynamic linker が `/nix/store/.../lib` 配下しか
見ない状態でビルドされており、システム標準の `/usr/lib/x86_64-linux-gnu/`
を参照しない。WSL2 上では GPU 用 `libcuda` は `/usr/lib/wsl/lib/` にあり、
これも素の Python からは見えない。

`scripts/env.sh` がランタイムで以下を `LD_LIBRARY_PATH` に追加して解決:

- `/nix/store/.../gcc-XX-lib/lib`  (libstdc++)
- `/nix/store/.../zlib-XX/lib`     (libz)
- `/usr/lib/wsl/lib`               (libcuda + 関連シム)

## How to apply

- Bash で python を呼ぶときは `scripts/run.sh python3 ...` を最初に試す
- `scripts/env.sh` を source して以降複数コマンドを動かすパターンも可:
  `. scripts/env.sh && python3 -m train.reinforce ...`
- pre-commit hook `check-main-py` は素の `python3` を使う (numpy 抜きの
  最小 import smoke なのでラッパー不要)
- torch wheel は CPU 版 (`pip install torch`) ではなく CUDA 版
  (`pip install --index-url https://download.pytorch.org/whl/cu128 torch`) を入れる
  — GPU が利用可能 (RTX 3070 Ti) なため。CPU 版を入れると `is_available()` が
  False を返して学習が遅くなる
- 新しい依存パッケージは venv に入れる (`.venv/bin/pip install ...`)。システム
  Python (`pip install --user`) は nix と衝突するので避ける
