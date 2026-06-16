---
description: cabt engine の知っておかないと事故る挙動 — litellm panic / Battle.battle_ptr orphan / obs None フェーズ
paths: "selfplay_test.py,train/reinforce.py,main.py,agent.py"
---

# kaggle_environments + cabt の罠は事前防御する

## 不変条件 1: litellm shim を kaggle_environments 前に install

```python
import sys
sys.modules.setdefault("litellm", type(sys)("litellm"))
from kaggle_environments import make   # ← ここで panic させない
```

理由: `kaggle_environments` を import すると werewolf env のロードで
`litellm` が要求され、それが `cryptography` の Rust 拡張を呼んで panic する。
Python の dummy module を `sys.modules` に先回りで突っ込むと werewolf env の
ロードは "failed" でログ出力されるだけで進行する。cabt env には影響なし。

この shim を ruff が E402 (top-level でない import) でエラーにするため、
`pyproject.toml` の `per-file-ignores` で `selfplay_test.py` と
`train/reinforce.py` に E402 を付与している。

## 不変条件 2: Battle.battle_ptr が None でないまま battle_start すると未定義動作

`make("cabt")` を毎エピソード呼び直せば C 側がリセットされる。
途中で例外が出てもエンジン側のハンドルは解放されない。長時間ループでは
`try/finally + make()` で必ず作り直す。

## 不変条件 3: obs["current"] と obs["select"] は None があり得る

```python
def agent(obs):
    sel = obs.get("select")
    if sel is None:
        return list(_DECK)          # ← deck 提出フェーズ
    # ここから select は dict 確定
```

- 初期デッキ提出: `select=None, current=None` → 60 枚の card ID 配列を返す
- 通常ターン: `current` は dict、`select` も dict

`obs["current"]` が None のとき `cur["players"]` を取りに行かない。
features 抽出側 (`train/features.py:state_features`) でも先頭で
`cur = obs.get("current")` してから `if cur is None: return zero_vec`。

## 不変条件 4: 戻り値の制約

`agent(obs)` の戻り値は `list[int]`:

- 単一選択 (`maxCount == 1`): `[chosen_index]`
- 複数選択: `minCount..maxCount` 個のインデックス配列、**重複禁止**
- `select is None`: card ID 60 枚配列 (インデックスではない!)

`random.sample(range(n), k)` は `k > n` で例外、`k > maxCount` で
engine が `IndexError` を投げる。

## How to apply

- 新規エージェント関数を書くときは `selfplay_test.py:random_agent` を
  雛形にする (select=None ハンドリング + sample がコンパクトに揃っている)
- 学習スクリプトが crash したあと再開するときは Python プロセス自体を
  再起動する (Battle.battle_ptr の orphan は同プロセス内に残る)
- features を拡張するときは必ず `cur is None` ガードを入れる
