# poke-ai

[PTCGABC / ポケカABC](https://ptcg-abc.pokemon.co.jp/) (Kaggle: [Pokémon TCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)) のシミュレーション部門に提出するエージェント。

`obs_dict` を受け取って `option` インデックスのリストを返す `agent.py` をベースに、numpy 線形ポリシーを self-play REINFORCE で学習させる構成。

## クイックスタート

```bash
# vs random ベンチ (~3秒 / 8試合)
python3 selfplay_test.py 4

# 自己対戦学習 (~3分 / 500ep)
python3 -m train.reinforce --episodes 500 --lr 0.05 --metrics-out train/metrics.json

# 学習済み重みは train/policy.npz に保存され、agent.py 起動時に自動ロード
```

## 構成

```
agent.py              Kaggle 提出エントリ。policy.npz があればロード、無ければ engine 順
selfplay_test.py      vs random ベンチ
train/
  features.py         state 36-d / option 36-d 特徴
  policy.py           numpy 線形ポリシー (.npz 保存/読込)
  reinforce.py        self-play REINFORCE
  policy.npz          学習済み重み
  metrics_*.json      学習履歴
NOTES.md              実機調査ログ (VERIFY 結果, obs スキーマ, OptionType 列挙)
CLAUDE.md             Claude Code 向け開発メモ
```

## 現状の強さ

| バージョン | vs random (16 戦) | 備考 |
|---|---|---|
| engine 順フォールバック | 14-2 | option index 0 を常に選ぶだけ |
| 500ep 学習 (state 24-d / opt 18-d) | 19-5 | PR #1 |
| 500ep + Pokemon-aware 特徴 (state 36-d / opt 36-d) | 23-1 | PR #2 |

## 詳細

- 実機調査 (cabt env の Python API が HANDOVER と異なる件、obs スキーマ、OptionType 列挙) は [`NOTES.md`](./NOTES.md)
- 開発フロー・罠・コミット規約は [`CLAUDE.md`](./CLAUDE.md)

## ライセンス

Pokémon / Nintendo / Creatures / GAME FREAK ほか各社の商標。
