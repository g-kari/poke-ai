# PPO 実装計画 — V60 振動制御の本命路線

未着手の design document。 V60 EXT1-EXT4 で観測された **policy gradient
variance による振動** (recent 勝率 0.17-0.25 で往復、 lr 制御では止まらず)
を解決する手段として、 PPO (Proximal Policy Optimization) の実装計画を
ここに残す。

## 1. なぜ PPO か (= V60 で動かなかった理由)

V60 REINFORCE での観測:

| metric | EXT1 (5500ep, lr=5e-4) | EXT3 (10500ep, lr=1e-4) | EXT4 (15500ep, lr=5e-5) |
|---|---|---|---|
| recent 勝率 | 0.17 - 0.25 | 0.17 - 0.25 | 0.21 - 0.26 |
| solo bench @ 30g | 20.1% | 20.5% | **13.3%** (谷を保存) |

**観察**: lr を 1/10 に下げても振動範囲は同じ。 これは **REINFORCE の policy
gradient sample variance** が支配的で、 lr では制御できない事を示す。

PPO の利点:
- **clipped surrogate objective**: ratio π_new(a)/π_old(a) を [1-ε, 1+ε]
  に clip → 1 batch で policy が暴れない
- **GAE (Generalized Advantage Estimation)**: 1-step / n-step advantage を
  λ で blend → variance vs bias の tunable trade-off
- **value baseline と policy gradient の separate optimization**:
  REINFORCE の `loss = -A * log_pi + (V - r)^2` 共用で value loss が policy
  を引き摺ったが、 PPO は 2 段階更新で分離

## 2. 設計の核

### 2.1 入出力

- features: features_v60.state_features (60-d) + features_v60.option_features (40-d)
  - V60 と互換性、 EXT3 を warm-start に使える
- policy network: MlpPolicyV60 そのまま (= pi: 64,32; v: 32)
- batch unit: episode = 1 game の trajectory
- batch size: 64 episodes per update (= GPU 1 step あたり ~3 分)

### 2.2 algorithm 擬似コード

```python
for iteration in range(N_iterations):
    # 1. Rollout: collect batch_size episodes from current policy
    trajectories = []
    for _ in range(batch_size):
        traj = run_episode_with_log_probs(policy, opponent_pool, rng)
        trajectories.append(traj)

    # 2. Compute GAE advantages and returns
    for traj in trajectories:
        traj.advantages = gae(traj.rewards, traj.values, gamma=0.99, lam=0.95)
        traj.returns = traj.values + traj.advantages

    # 3. PPO updates: k_epochs × minibatch over the batch
    for epoch in range(k_epochs):  # 4-10
        for mb in minibatches(trajectories, mb_size=32):
            # ratio = exp(new_logp - old_logp)
            ratio = torch.exp(policy.log_prob(mb.obs, mb.actions) - mb.old_log_probs)

            # clipped surrogate loss
            surr1 = ratio * mb.advantages
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * mb.advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # value loss (separate)
            v_pred = policy.value(mb.obs)
            value_loss = (v_pred - mb.returns).pow(2).mean()

            # entropy bonus for exploration
            entropy = -policy.probs(mb.obs).log().sum(dim=-1).mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
```

### 2.3 hyperparameters (= V60 EXT3 から warm-start 前提)

| name | value | 理由 |
|---|---|---|
| `lr` | 3e-5 | warm-start 後の細かい調整、 REINFORCE 1e-4 の 1/3 |
| `gamma` | 0.99 | 30+ ターンの bridge、 終局報酬の discount |
| `lam` | 0.95 | GAE smoothness |
| `epsilon` | 0.2 | clip 標準値 |
| `k_epochs` | 4 | minibatch 再利用回数 |
| `batch_size` | 64 episodes | GPU memory 制約 |
| `mb_size` | 32 | gradient noise vs throughput |
| `entropy_coef` | 0.01 | exploration、 collapse 防止 |
| `value_coef` | 0.5 | policy vs value loss balance |
| `clip_grad` | 0.5 | gradient norm cap |
| `total_iterations` | 200 | = 200 × 64 = 12800 episodes ≈ V60 EXT3 同等 |

### 2.4 期待される改善

V60 EXT3 (REINFORCE) との比較で:

| metric | EXT3 REINFORCE | PPO 推定 |
|---|---|---|
| recent 勝率 | 0.17-0.25 (振動) | 0.20-0.28 (収束) |
| solo bench @ 80g | 19.7% | **23-26% 期待** |
| LB 推定 | 523 | **680-800 期待** |

理由:
- variance 削減で hard matchup (Crustle Dashi 3.8%) でも policy update が
  進む
- clipped objective で 1-batch の暴走を防ぎ、 振動を抑制
- GAE で credit assignment 改善 (= 30 ターン後の報酬を中間 action に
  正しく分配)

## 3. 実装の段階分け

### Phase 1: replay buffer + log-prob 記録 (1 サイクル)

REINFORCE 同様 trajectory を貯めるが、 各 step で:
- state features
- action index
- log π(action | state) ← 重要、 PPO の ratio 計算に必要
- value V(state)
- reward (1 game = 最後だけ ±1)

`train/replay_buffer_v60.py` に `EpisodeBuffer` を実装。

### Phase 2: GAE advantage 計算 (1 サイクル)

```python
def gae(rewards, values, gamma, lam):
    advantages = []
    last_advantage = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        last_advantage = delta + gamma * lam * last_advantage
        advantages.insert(0, last_advantage)
    return advantages
```

1 game = 1 reward (終局のみ) なので 中間 step は reward=0 で GAE は
value bootstrap になる。

### Phase 3: PPO loss + update (1 サイクル)

clipped surrogate + value MSE + entropy。 既存 `mlp_train_v60.py` の
`reinforce_update` を `ppo_update` に置換。

### Phase 4: warm-start from EXT3 (1 サイクル)

`train/mlp_policy_v60_ext3.pt` を初期 weights に、 5000-10000 PPO
iterations (= 1-2 日 GPU)。

### Phase 5: bench + submission (1 サイクル)

`scripts/bench_v60.py` で評価。 25%+ なら EXT3 と入替えで再 submit、
LB 700+ を狙う。

## 4. リスク

1. **PPO でも variance 残る**: PTCG の sparse reward (= ±1 終局のみ) で
   GAE bootstrap が短かすぎる可能性。 reward shaping (= 中間で prize-delta
   報酬) を加える必要があるかも。
2. **EXT3 warm-start が PPO の前提を破る**: REINFORCE で学習した policy
   は GAE の value 推定が外れていて、 初期 PPO updates で大きく動く。
   value head だけ pre-train するか、 PPO 初期に lr を更に下げる。
3. **40 分 × 1 日学習で improvement が出ない**: PPO の rule of thumb は
   100k-1M episodes、 我々の 12800 では足りない可能性。

## 5. 後継者への引き継ぎ事項

このファイルを読む人へ:

1. **既存資産**:
   - `train/features_v60.py` (60-d features)
   - `train/mlp_policy_v60.py` (MLP + value head)
   - `train/mlp_train_v60.py` (REINFORCE 訓練ループ)
   - `train/mlp_policy_v60_ext3.pt` (10500ep weight、 warm-start に使える)
   - `scripts/bench_v60.py` (評価 CLI)

2. **避けるべき罠**:
   - main_v60.py の **deck.csv path resolution** は 3-MLP main.py パターン
     (= `"deck.csv"` → `"/kaggle_simulations/agent/deck.csv"` の 2 段階
     fallback) を継承すること。 53810836 / 53812115 ERROR の再発を防ぐ
   - bundle に **`scripts/extract_v60_weights.py` + .npz** を含めて
     torch 依存を除去すること (Kaggle ランタイムには torch なし)
   - submit 前に `check_main_exec.py --strict-cwd --no-policy` で
     **Kaggle-strict simulation** で deck.csv path 解決を確認

3. **PPO 以外の中長期 path** (= 同 priority):
   - **AlphaZero**: PIMC + 学習 value head (= PIMC 文脈で self-distillation)
   - **Behavioral cloning from V6**: V6 agent の選択を v60 features 上で
     supervised learning、 V6 90% 模倣を狙う

---

最後に: PPO 実装は **大規模 refactor** で 5-10 サイクル分の投資を要する。
シミュレーション部門 (締切 8/16) には間に合わない見込みだが、 ストラテジー
部門 (締切 9/14) で「未着手だが完全な設計書」 として report 添付可能。
