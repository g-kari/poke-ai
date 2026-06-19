# AlphaZero 実装計画 — PIMC 棄却後の本命路線

未着手の design document。 PIMC v1-v6 で **rollout policy の質と無関係に
tie** (= 補遺 15-17) が確定し、 PIMC アーキテクチャ自体が limit。 ここから
先に single policy 性能を上げる手段として AlphaZero (MCTS + value head)
の実装計画を残す。

## 1. なぜ AlphaZero か (= PIMC が動かなかった 3 つの理由を解決)

PIMC v1-v6 で観測した limit:

1. **1-ply lookahead の浅さ**: 局所最適化が opp の最適応答に脆弱
2. **Naive opp_state sampling**: 相手 deck = 我々と仮定する prior が
   実際の opp 分布と乖離
3. **Rollout-to-terminal noise**: 終局 reward ±1 で high variance、
   N_worlds=2 では平均化不足

AlphaZero はこれを構造的に解決:

- **Deep MCTS (depth N)**: 1-ply ではなく N 手先まで search、 PUCT で
  exploration / exploitation balance
- **Learnt value head**: rollout-to-terminal の代わりに NN が
  state → value を予測、 noise を学習で吸収
- **Opp prior learning**: self-play で opp 行動分布を policy が学習、
  naive sampling 不要

我々の compute 制約下では完全 AlphaZero (= 10^8 self-play games) は無理
だが、 **AlphaZero 縮小版 (= PPO_v40 base + value head + 浅 MCTS)** で
PIMC 失敗の 3 原因を緩和できる。

## 2. 設計の核 — 3 構成要素

```python
class AlphaZeroAgent:
    def __init__(self, policy_net, value_net, mcts_simulations=50):
        self.policy = policy_net  # P(action | state) — 既存 PPO_v40 s500
        self.value = value_net    # V(state) — 新規学習が必要
        self.n_sims = mcts_simulations

    def choose(self, obs):
        root = MCTSNode(obs)
        for _ in range(self.n_sims):
            leaf = self.tree_policy(root)  # PUCT で leaf 選択
            value = self.value(leaf.obs)   # value head で評価
            leaf.backup(value)             # 結果を root まで伝播
        return root.argmax_visit()         # 最頻 visit option
```

**3 構成要素**:

### 2.1 既存 PPO_v40 policy をそのまま流用

- `train/mlp_policy_ppo_v40_s100_t500.pt` (= Mode B、 補遺 10 で 3-MLP
  base を 77.5% で勝つ強い policy) を policy_net に
- PUCT の prior として `policy.probs(obs, sel)` を直接使う

### 2.2 Value head の学習

新規学習が必要:
- **入力**: state features (= 40-d v40 か 60-d v60、 sweep で決定)
- **出力**: V(state) ∈ [-1, +1] (= 終局報酬の期待値)
- **教師データ**: 既存 PPO 訓練ログ (= 終局 reward × 各 step の state)
  または self-play で新たに生成
- **アーキテクチャ**: 64-32 MLP (= 既存 mlp_policy と同サイズ)
- **損失**: MSE(V(s), 終局報酬)

### 2.3 MCTS (= 浅い PUCT)

```python
class MCTSNode:
    def __init__(self, obs, prior_p):
        self.obs = obs
        self.P = prior_p          # policy.probs から
        self.N = 0                # visit count
        self.W = 0.0              # total value
        self.Q = 0.0              # mean value W/N
        self.children = {}        # option_idx -> MCTSNode

    def select_child(self, c_puct=1.4):
        """PUCT formula: Q + c * P * sqrt(parent.N) / (1 + N)"""
        best = max(self.children.items(),
                   key=lambda kv: kv[1].Q + c_puct * kv[1].P *
                                  math.sqrt(self.N) / (1 + kv[1].N))
        return best
```

**PUCT パラメータ**:
- `c_puct=1.4` (= AlphaGo Zero 標準)
- `n_simulations=50` (= Kaggle 5s budget で実現可能)
- `temperature=0` (= 推論時は argmax visit、 学習時は softmax)

## 3. アルゴリズム — Phase 1 (= minimum)

```python
def alphazero_choose(obs, policy, value_head, n_sims=50):
    """1 つの観測に対して MCTS で option を選択."""
    sel = obs["select"]
    if sel is None or sel.get("type") != 0 or int(sel.get("maxCount") or 0) != 1:
        return None  # fallback to policy

    opts = sel["option"]
    if len(opts) <= 1:
        return [0] if opts else []

    # Root expansion
    root_priors = policy.probs(obs, sel)
    root = MCTSNode(obs, prior_p=1.0)
    for i in range(len(opts)):
        root.children[i] = MCTSNode(None, prior_p=root_priors[i])

    # MCTS simulations
    for _ in range(n_sims):
        leaf = root
        path = [leaf]

        # 1. Selection — PUCT 降下
        while leaf.children and leaf.obs is not None:
            i, leaf = leaf.select_child()
            path.append(leaf)

        # 2. Expansion — leaf を search_step で展開
        if leaf.obs is None:
            try:
                leaf_obs = expand_leaf_via_search(obs, leaf.action_path)
                leaf.obs = leaf_obs
                # Init priors and value
                leaf_priors = policy.probs(leaf_obs, leaf_obs["select"])
                for j in range(len(leaf_obs["select"]["option"])):
                    leaf.children[j] = MCTSNode(None, prior_p=leaf_priors[j])
                value = value_head(leaf_obs)
            except Exception:
                value = 0.0
        else:
            value = value_head(leaf.obs)

        # 3. Backup
        for node in reversed(path):
            node.N += 1
            node.W += value
            node.Q = node.W / node.N

    # Return most-visited child
    return [max(root.children.items(), key=lambda kv: kv[1].N)[0]]
```

## 4. 計算予算 — Kaggle タイムアウト

Kaggle 1 ターン制限を 5 秒と仮定:
- value head forward: ~5 ms (small MLP)
- search_step + sample_opp_state: ~30 ms (= PIMC v5 の 50ms より速い、
  rollout-to-terminal 不要のため)
- 1 simulation = ~35 ms → **5 秒で ~140 simulations 可能**
- 実用的には n_sims=50 で ~1.75 秒 = safety margin あり

**比較**:
- PIMC v5: 1 call = 196 ms (= 2 simulations × 80-step rollout)
- AlphaZero v1: 1 call = 1.75 秒 (= 50 simulations × no rollout)
- 計算量 9 倍だが、 simulations は 25 倍

## 5. 期待される lab 改善

n_simulations と期待 lab (= 我々の PPO_v40 s500 base 18.6% から):

| n_sims | 期待 lab | 期待 LB (ratio 35) | 計算コスト |
|--------|----------|--------------------|------------|
| 1 (= policy のみ) | 18.6% | 651 | 1× |
| 10 | ~20% | 700 | 5× |
| 50 | ~22-25% | 770-875 | 25× |
| 200 | ~25-28% | 875-980 | 100× |

AlphaGo MC tree search の lift 効果から外挿。 ただし value head の質に
強く依存 (= 値が unreliable なら lift しない)。

## 6. 実装フェーズ (= 9/14 STRATEGY 締切まで)

- **Phase 1** (2-3 サイクル): value head の単体実装 + 学習
  (`train/value_head.py`、 教師データ = 既存 PPO 訓練 episodes)
- **Phase 2** (2 サイクル): MCTSNode + alphazero_choose 実装
  (`train/alphazero.py`)
- **Phase 3** (1 サイクル): main.py 統合 (= POKEAI_ALPHAZERO=1 環境変数)
- **Phase 4** (2-3 サイクル): n_sims sweep (10, 50, 200) + 提出 bundle
  作成
- **Phase 5** (1 サイクル): mirror match で PIMC-OFF vs AlphaZero-ON、
  ≥ 60% winrate なら default-ON 化、 Kaggle 提出

## 7. 着手判断 (= 明日 UTC LB 着地後)

- **LB ≥ 815** (= s100/s2026 で ratio 35 達成): single policy 路線が
  すでに十分強い、 AlphaZero は **更なる改善路線** として 9/14 までに
  実装
- **LB 700-800**: single policy 天井 ratio 30、 AlphaZero で **天井
  突破の本命路線**
- **LB < 700**: 根本的な再設計が必要、 AlphaZero よりも先に features
  改善 or deck 切替を検討

## 8. リスク評価

**成功確率**: 中-高
- AlphaGo Zero の方法論は確立、 縮小版でも lift する可能性高い
- ただし value head の学習が成否を決める = 教師データの質が key

**失敗時のフォールバック**:
- もし AlphaZero でも tie (= PIMC v6 同様) なら、 PTCG ABC は探索が
  本質的に効かない game (= 情報集合が広すぎる) ことを示唆
- その場合は **features 根本見直し** (= card-level embedding) or
  **deck 戦略の構造的変更** (= meta-deck 制覇路線) に path 移行

## 9. 参考

- AlphaGo Zero paper (Silver et al. 2017): PUCT + value head + self-play
- MuZero (Schrittwieser et al. 2020): model-based、 不完全情報でも動く
  variant、 PTCG ABC のような hidden info game に直接適用可能
- 我々の `cg/api.py` の search_begin/step は既に PIMC で使用済、
  AlphaZero の expansion step も同じ API で動く
