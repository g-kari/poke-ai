# PIMC 実装計画 — PPO 天井突破の次手

未着手の design document。 PPO_v40 chain で n=4 試行 + 2 ext + 4 ensemble
全て試行済み、 lab 23% 級は **ガチャで偶然引いた解** で 2nd stage では
必ず劣化することが [[STRATEGY_REPORT 5.3l]] で確定。 ここから先に
single policy 性能を上げる手段は **探索付き推論 (= search-augmented
inference)** しかない。 ここに PIMC (Perfect Information Monte Carlo)
の実装計画を残す。

## 1. なぜ PIMC か (= PPO の天井理由)

PPO で観測した limit (= 5.3l 補遺 7 まで):

- **PPO seed ガチャ**: 同じ base + lr + entropy で random seed のみ
  変えるだけで lab 16.3% (seed=42) ~ 23.3% (seed=0) と ±7pp の幅
- **2nd stage 必ず劣化**: 全 PEAK 解 (s100, s2026) を warm-start で
  延長すると -3.4 ~ -3.5pp の regression、 specialization profile は壊れる
- **Ensemble 全パターン失敗**: features 起因 / training 起因 / strength
  不均衡 / specialization 境界 — 4 失敗パターン分類で完全否定

含意: **静的 policy (= 入力 → 出力 の 1 回 forward) の lab 天井は ~23%**。
これ以上は同じ policy を別の使い方をするしかなく、 search が唯一の道。

PIMC は **inference 時に各候補手を Monte Carlo で展開** して、 終局
報酬の期待値が最大の手を選ぶ。 利点:

- **学習不要**: 既存 PPO policy を rollout policy にそのまま流用
- **探索深さで lab が monotone に伸びる** 可能性 (= alpha-zero 級は
  search 1000 で +20pp 改善した例 [AlphaGo 論文])
- **ratio 35 仮説下で lab 30% → LB 1050** が見える: PPO 天井 23% を
  search で +7pp 持ち上げれば 3-MLP base LB 679 を遥かに超える

## 2. 設計の核 — cg.api 公式 PIMC API

`cg/api.py` (vendor 済み) が search 関数を公式 expose:

```python
# 探索開始 — 相手の隠れ情報 (deck, hand, prize, active) を引数で渡す
search_begin(
    agent_observation,  # 我々の obs (= search_begin_input 必須)
    your_deck,          # 残りデッキ card ID list
    your_prize,         # prize card ID list
    opponent_deck,      # 相手デッキ予測 (= 後述 information set sampling)
    opponent_prize,     # 相手 prize 予測
    opponent_hand,      # 相手手札予測
    opponent_active,    # 相手の裏 active 予測 (face-down のみ必要)
) -> SearchState

# 探索ステップ — option を 1 つ選んで進める
search_step(search_id, select) -> SearchState

# 探索終了 — メモリ解放
search_release(search_id)
```

戻り値の `SearchState` には新しい `Observation` + `reward` + 終局 flag が
含まれる。 これを使って rollout を実装する。

## 3. アルゴリズム (= PIMC v1 minimum)

```
def pimc_choose(obs, policy_fn, n_samples=20, max_depth=20):
    """
    obs: 我々の current observation (= main.py の agent(obs) の引数)
    policy_fn: lambda obs, sel → list[int]  (= 既存 PPO policy)
    n_samples: 各 root option で何回 rollout するか
    max_depth: rollout 深さの上限 (game length は ~30 ターン)

    return: 最大 Q の option index
    """
    sel = obs["select"]
    n_opts = len(sel["option"])
    Q = [0.0] * n_opts

    for i in range(n_opts):
        for _ in range(n_samples):
            # 1. 相手の隠れ情報を sampling
            opp_deck, opp_hand, opp_prize, opp_active = sample_opp_state(obs)

            # 2. search_begin で root を開く
            state = search_begin(obs, our_deck, our_prize,
                                 opp_deck, opp_prize, opp_hand, opp_active)
            search_id = state.search_id

            # 3. root option i を選択
            state = search_step(search_id, [i])

            # 4. 終局まで両 player を policy_fn で plays
            depth = 0
            while not state.is_terminal and depth < max_depth:
                next_sel = state.observation["select"]
                action = policy_fn(state.observation, next_sel)
                state = search_step(search_id, action)
                depth += 1

            # 5. 終局報酬を Q に加算
            Q[i] += state.reward
            search_release(search_id)

        Q[i] /= n_samples

    return int(np.argmax(Q))
```

**注**: 上記は素朴な PIMC v1。 IS-MCTS (Information Set MCTS) で UCB1
を使うと探索効率が大きく改善するが、 v1 で動くこと優先。

## 4. Information Set Sampling — 最大の難所

相手の隠れ情報は完全には推定できない。 v1 では **naive sampling**:

```python
def sample_opp_state(obs):
    # 4-a. opp deck: 自分のデッキと同じと仮定 (= naive)
    opp_deck = list(DECK)  # 60 cards

    # 4-b. visible cards (discard, bench, active, prize) を全部除外
    opp_visible = set()
    opp_state = obs["current"]["players"][1 - obs["current"]["yourIndex"]]
    for area in [opp_state["discard"], opp_state["bench"], opp_state["active"]]:
        for card in area:
            if card is not None:  # face-down は除外
                opp_visible.add(card["id"])

    # 4-c. hand_count に対して残りからランダム sample
    available = [c for c in opp_deck if c not in opp_visible]
    np.random.shuffle(available)
    hand_count = opp_state["handCount"]
    opp_hand = available[:hand_count]
    opp_deck = available[hand_count:]

    # 4-d. prize は 6 枚を残りから sample
    prize_count = len(opp_state["prize"])
    opp_prize = opp_deck[:prize_count]
    opp_deck = opp_deck[prize_count:]

    # 4-e. face-down active (= 例: 1st turn の裏ポケモン) は basic Pokémon を sample
    opp_active = []
    active = opp_state["active"]
    if active and active[0] is None:
        # 残りから basic Pokémon を 1 枚 sample
        opp_active = [pick_basic_pokemon(available)]

    return opp_deck, opp_hand, opp_prize, opp_active
```

**改善方向 (v2 以降)**:
- 相手のデッキ予測を **rule-based pool で観察した meta-deck の prior**
  から sampling (= 4 つの rule_based agent の deck.csv を pool 化)
- Bayesian update: 相手の play から hand 内 card 分布を絞り込む
- belief state: opponent_hand を particle filter で追跡

## 5. 計算予算 — Kaggle タイムアウト制約

Kaggle ランタイムの 1 ターン制限は **5 秒** (公式不明確、 仮定)。 PIMC の
1 rollout = ~20 step × policy forward (= 数 ms) = ~100 ms。

- n_samples=20, n_opts=10 → 20×10 = 200 rollouts = 20 秒 — **超過**
- n_samples=5, n_opts=5 → 25 rollouts = 2.5 秒 — OK
- v1 では **n_samples=3, n_opts は MAIN single-choice のみ** に絞る

## 6. 期待される lab 改善

n_samples ごとの lab 期待値 (= 我々の PPO_v40 seed=100 base 23.3% から):

| n_samples | 期待 lab | 期待 LB (ratio 35) | コスト |
|-----------|----------|--------------------|--------|
| 1 (= policy のみ) | 23.3% | 815 | 1× |
| 3 | ~25% | 875 | 3× |
| 5 | ~27% | 945 | 5× |
| 10 | ~29% | 1015 | 10× |

期待値は AlphaGo の MC tree search 効果から外挿。 実測で確認必要。

## 7. 実装フェーズ

- **Phase 1** (1-2 サイクル): cg.api.search_* の動作確認、 sample_opp_state
  の単体テスト
- **Phase 2** (2-3 サイクル): pimc_choose minimum 実装、 selfplay_test で
  動作確認、 単純 rollout の lab 測定
- **Phase 3** (1 サイクル): main.py への統合、 PIMC 入りの提出 bundle 作成
- **Phase 4** (2 サイクル): タイムアウト調整、 n_samples sweep、 best
  バリアントの提出

## 8. 着手判断 (= 明日 LB 着地点後)

- **LB ≥ 815** (= ratio 35 確認): PPO seed=100 / seed=2026 のままで LB
  815 維持、 PIMC は更なる改善路線として価値あり
- **LB 700-800** (= ratio 30-34): PPO 天井は ratio 30 で頭打ち、 PIMC で
  search 効果を確認する価値が高い
- **LB < 700** (= overfit シナリオ): PPO の 23% は LB に translate せず、
  根本的な再設計が必要 — PIMC 着手前に lab metric の見直し

## 9. 参考

- AlphaGo paper (Silver et al. 2016): MCTS + policy network で +10% 強さ
- Tron AI (= 古典 PIMC): n_samples=1000 で perfect-info AI に勝てる
- 我々の cg/api.py は公式 PIMC support なので、 実装難易度は低い (1 週間)
