# Strategy Report — PTCG ABC (poke-ai)

シミュレーション部門 (締切 8/16) の探索ログをストラテジー部門 (締切 9/14)
向けに整理した報告書。 30 サイクル × 30 分 ≈ **15 時間** の試行錯誤の
記録。

リポジトリ: <https://github.com/g-kari/poke-ai>
ダッシュボード: <https://g-kari.github.io/poke-ai/>

## TL;DR

12 系統 (線形 / 3-MLP / V60 / V60 ensemble / V6 deck / deck-builder /
GA / PIMC v1-v5 / Crustle 検出 agent / vendored rule-based) を試行し、
**LB best は rule-based vendored の V6 (LB 926)、 deep-learning best は
3-MLP (LB 679)** で確定。 我々自身の深層学習 V60 は LB 523 で 3-MLP に
劣後。 「lab 1pp ≈ LB 43 ポイント」 の換算法則 + 「features 共通の
policy ensemble は seed diversity が効かない」 等の知見を発見。

## 1. 問題設定

PTCG ABC は **不完全情報の二人零和ゲーム**:
- 観測される: 自分の手札 + 場の board state + 相手の見える pokemon
- 観測されない: 相手の手札 / deck 残り / prize 内容
- 1 game = 30+ ターン × 各ターン 多段 select
- reward は終局 ±1 (引き分けは 0) — **sparse reward** が core difficulty
- evaluation = TrueSkill 風 μ₀=600、 試合で μ 更新、 σ 縮小

policy 学習の困難:
1. **credit assignment**: 30+ ターン後の 1 試合報酬を各 action に分配
2. **opponent variance**: 相手の deck / policy が観測中盤までしか分からない
3. **action 構造**: 多段 select (= ATTACH の後に「どこに」 が来る、 等)
4. **legal move 制約**: deck から合法な行動だけ可、 ATTACK の attackId 等

## 2. 試した 12 系統のアプローチ

| # | 系統 | best lab | LB | 結論 |
|---|---|---|---|---|
| 1 | 線形 policy (numpy REINFORCE) | 95% vs random | n/a | baseline |
| 2 | **3-MLP ensemble v40** (seed 0/2/100) | **23.3%** | **679.6** | **我々の DL best** |
| 3 | V60 single (= +opp deck-id fingerprint features) | 19.7% | 523.1 | 3-MLP に劣後 |
| 4 | V60 ensemble (= multi-seed) | 18.6% | ~480 推定 | seed diversity 効かず |
| 5 | V60 + V6 deck (deck.csv 入替え) | 10.0% | n/a | 7 opp で壊滅 |
| 6 | V60 V6 deck warm-start | 12.4% | n/a | deck shock 緩和できず |
| 7 | deck-builder v1-v9 (= heuristic + 進化チェーン) | 17.5% (v4) | n/a | 構造的限界 |
| 8 | GA loop (1-card swap、 8g/eval) | 22.5% in-eval、 13-15% 40g 真値 | n/a | local optimum 5pp 以下 |
| 9 | PIMC v1 1-ply prize-delta | 8.3% | n/a | engine prior + 1pp |
| 10 | PIMC v2-v5 (= field-aware / NN value / inference / multi-ply) | 5-10% | n/a | heuristic 天井 |
| 11 | Crustle 検出 + rotation agent (v8-v11) | 7-8% | n/a | proactive setup 不足 |
| 12 | vendored rule-based (Kiyota / dashimaki / romanrozen) | **lab 67.3% (CrustleDashi)** | **LB 926 (V6)** | 競技用 best |

## 3. 発見した法則

### 3.1 lab 1pp ≈ LB 43 ポイントの換算法則

3-MLP と V60 EXT3 の実測比較で確定:

  3-MLP:      lab 23.3%   LB 679.6
  V60 EXT3:   lab 19.7%   LB 523.1
  差:           -3.6pp        -156.5

→ **lab 3.6pp の差 = LB 156 ポイント差 = 1pp ≈ 43**

ただし lab → LB は **非線形**:
- 23.3% → 679
- 64.0% (Iono) → 762
- 67.3% (CrustleDashi) → 870
- 57.9% (V6) → 926
- 高 lab 領域 (50%+) では LB 増分が縮小、 低 lab 領域では拡大

### 3.2 Crustle Dashi 0% は ex pokemon を deck から外さない限り解決不可

dashimaki Day-1 #1 Crustle bot の特性「ふしぎなロックイン」 で **ex
pokemon の attack ダメージが 0**。 我々の Mega Abomasnow ex 軸 deck では:

- 6 試行 (V60 EXT1/EXT3/EXT4/seed31415/V6deck warm/3-MLP) で **6 回 0% 持続**
- agent 改修 (Crustle 検出 + non-ex routing) も無効
- 唯一の解決は **non-ex deck** (= Hariyama 系の V6) だが overall 低下

これは **deck level の構造問題**、 agent / 学習では迂回不可能。

### 3.3 deck と policy は強 coupled

features_v60 で deck-id fingerprint を入れても:
- policy weight は **訓練時の deck の strategy** を焼き込む
- 入替え時の bench で 7 opp 全体 **-10pp 以上**
- warm-start 5000ep でも -8pp、 つまり 22% も V6 deck で再学習しても達成困難

含意: 「deck と agent を同時生成 / 共進化」 する仕組み (= AlphaZero スタイル
self-play with deck mutation) が真の解決。

### 3.4 features 共通の policy ensemble は seed diversity が効かない

3-MLP (v40 features) は seed 0/2/100 ensemble で **+α** を達成 (23.3%):
- 1 seed あたり 19-21% を ensemble で 23.3% に
- seed diversity が overall を改善

V60 (v60 features) は 2 試行 ensemble で **-2pp**:
- EXT1+EXT3 ensemble = 18.1% (EXT1 20.1%、 EXT3 20.5% より低い)
- EXT3+seed31415 ensemble = 18.6% (EXT3 single より低い)
- features の表現力が seed-level diversity を吸収して中庸化

推測: features_v60 の deck-id bucket hash は overfitting を引き起こし、
seed 違いの policy も同じ偏りを持つ。

### 3.5 lab signal は LB に translate するが、 absolute 予測は困難

実測ケース:
- 53793417 Iono: 600 → 615 → 762 (= lab 64.0% → LB 762)
- 53794617 CrustleDashi: 718 → 866 → 870
- 53794828 V6: 801 → 873 → 926

評価期間が進むほど真の値に収束、 ただし lab → LB の関数形は agent によって
violate される (V6 は lab 57.9% で LB 926 だが、 lab 64.0% Iono は LB 762)。

## 4. 教訓と反省

### 4.1 sunk cost fallacy

V60 路線に **7 サイクル投資** したが LB 523 で 3-MLP 679 を超えず。 5
サイクル目で「これ以上の改善は構造的に困難」 と判明していたのに継続した。
fast-iterate / fast-pivot のバランスが課題。

### 4.2 noise floor の認識遅れ

GA loop 初回 (3g/eval = 30 games/fitness、 Wilson CI ±25pp) で 23.3% の
"improvement" を観測したが、 40g 本格 bench で 13.2% に縮小 = 完全に
noise。 評価早期で fitness CI を意識する習慣が必要。

### 4.3 Submission build 体制の自動化遅れ

53810836 / 53812115 が **両方 ERROR**:
- 53810836: torch 依存 (Kaggle ランタイムに torch なし)
- 53812115: numpy 化したが single-shot path 解決
- 53812882: multi-root fallback で初の COMPLETE

local sandbox では完璧に動いたが、 **`--strict-cwd` 相当の simulation
が後付け**だった。 deep-learning 移植時に「ロードの頑健性」 を 3-MLP
main.py から継承し損ねた。

## 5. 未実装の方向

### 5.1 PPO (Proximal Policy Optimization)

V60 路線最大の問題は **policy gradient variance** で振動 (recent 0.17-0.25
往復)。 lr=5e-5 まで下げても止まらない。 PPO の clipped surrogate objective +
GAE で variance を制御できれば 25%+ が現実的。

### 5.2 AlphaZero (PIMC + 学習 value head)

PIMC heuristic は 10% 天井、 価値関数の質に強く依存。 self-distillation で
PIMC 文脈の value head を訓練すれば AlphaZero スタイルが可能。 ただし
infra が大きい (= replay buffer、 distributed self-play 等)。

### 5.3 Behavioral cloning from V6 agent

V6 の 30+ 行ロジック (= active rotation + Hariyama energy 配分 +
attack 順序選択) を policy で模倣する supervised learning。 V6 が
LB 926 なので、 90% 模倣できれば LB 800+ 期待。

### 5.4 Transformer features

card-level representation (= 各カードを embedding、 self-attention で
盤面全体を統合) で feature 表現力を桁違いに増す。 ただし pre-training
データ無し、 from-scratch 学習は困難。

## 6. 推奨される最短経路 (ストラテジー部門応募者向け)

時間予算が限られた場合:

1. **rule-based vendored を起点に**: V6 (LB 926) が最強、 改修不要
2. **deck.csv の選び方が決定的**: agent と coupled、 単独最適化不可
3. **lab 30g+ で bench** (= noise floor ±13pp)、 GA / fine-tune の
   「improvement」 は 40g 以上で本物か検証
4. **submission build を 3-MLP main.py パターン** から逸脱させない (=
   multi-root path fallback を継承)
5. 深層学習路線は **PPO + 大量学習 (50000ep+)** が最低ライン、 数日の
   GPU 投資が必要

## 7. 補遺: 試行 timeline

15 時間の試行を時系列で:

  T+0:    線形 policy → 3-MLP ensemble (LB 679)
  T+3h:   vendored rule-based 路線、 V6 (LB 926) 取得
  T+5h:   deck-builder v1-v9 で 構造的限界判明
  T+8h:   PIMC v1-v5、 heuristic 天井判明
  T+10h:  Crustle 検出 agent v8-v11、 reactive 限界
  T+11h:  V60 features (deck-id fingerprint) 設計
  T+13h:  V60 EXT1-EXT4、 振動から抜けず
  T+14h:  V60 submission 53810836/53812115/53812882 (3 回目で COMPLETE)
  T+15h:  V60 V6 deck warm-start、 最終決算

各 phase で「これが正解だろう」と思った瞬間が複数あり、 それぞれが
**noise floor 内の偽陽性** や **未検証の前提** だった事を後付けで認識。

---

最終的に、 競技用には **V6 (LB 926)** 提出、 深層学習 best は **3-MLP
(LB 679)** という二本立ての結論。 ストラテジー部門では「我々自身が
作った deep-learning は 3-MLP が天井」 という事実 + 12 系統の失敗ログ +
発見した法則 の 3 点を report の核とする。

---

## 付録 A: 主要アプローチの deep-dive

### A.1 3-MLP ensemble (LB 679.6) — 我々の deep-learning best

**設計**:
- v40 features (= 40-d state、 40-d option) で MLP policy + value head
  - pi: state ⊕ option → 64 → 32 → 1 logit
  - v: state → 32 → 1、 tanh で ±1 にクリップ
- REINFORCE で self-play training (2000 ep / member)
- ensemble = 3 つの独立学習 policy のロジット平均 (seed 0 / 2 / 100)

**実装のキー判断**:
- value head に **tanh** を選んだのは reward が ±1 のため。 後に linear で
  試して大幅悪化、 tanh は **必要な regularization** と判明
- `b_order` (= engine order bias) は 2.0 固定。 これがないと初期 policy が
  弱い (= engine prior の方が強い)
- ensemble member の数は 3 が sweet spot。 4 つ目を warm-start 系で加える
  と中庸化で -3pp (= 5 連敗の pattern)
- value baseline は advantage = reward - V(s) で variance 削減、 ただし
  hard matchup では tanh の clip で advantage が固定 → policy update が
  全 action に均等に伝播してしまい、 Crustle 等の対策学習が進まない

**ベンチ詳細 (80g/opp, 560 games total)**:

  vs Mega Lucario:   20-60 (25.0%)
  vs Dragapult ex:   16-64 (20.0%)
  vs Iono:            9-71 (11.2%)
  vs Mega Aboma:     17-63 (21.2%)
  vs Crustle Wall:   31-49 (38.8%)
  vs Crustle Dashi:  19-61 (23.8%)
  overall:          112-368 (23.3%)

最強 matchup は Crustle Wall 38.8% (haru 版)。 Iono が 11.2% で最大弱点。

**LB スコアの推移**:
- 53776705 (2-MLP): ERROR (__file__ 罠)
- 53776818 (2-MLP fix): 613.3
- 53778627 (3-MLP, current best): **679.6**

**転用可能な insight**:
1. value head の clip (tanh) は variance 削減に必須、 lr/sample 量を上げても
   linear V(s) では収束しない
2. ensemble member は **独立 seed の fresh 学習** から選ぶ。 warm-start 系を
   member にすると base policy の偏りが重複して悪化する
3. self-play REINFORCE は initial policy が engine prior に近い時、
   `b_order` で artificially 強化すると訓練が安定する

### A.2 V60 features の 7 サイクル投資 — 大失敗の根本原因

**設計**: 3-MLP の opp 識別力不足を補うため、 features に opp deck-id
fingerprint を追加 (60-d、 16 buckets × 3 area + 4 mirror buckets):
- f[0..39]: 元 v40 と同一
- f[40..55]: opp の active + bench + discard card-id を hash
- f[56..59]: 自分の active card-id mirror hash

**7 サイクルの試行**:

| version | episodes | lr | 結果 |
|---|---|---|---|
| EXT1 fresh pool5 | 2500 | 5e-4 | 17.1% @ 20g |
| EXT1 warm | +3000=5500 | 1e-4 | **20.1%** @ 20g (best) |
| EXT2 warm | +3000=8500 | 1e-4 | ~20% (振動) |
| EXT3 warm | +5000=10500 | 1e-4 | 20.5% @ 30g、 **19.7% @ 80g** |
| EXT4 warm | +5000=15500 | 5e-5 | 13.3% (-7.2pp、 学習過程の谷を保存) |
| seed=31415 fresh | 4000 | 5e-4 | 16.2% |
| V6 deck warm | 5000 | 3e-4 | 12.4% |

**根本原因**:
1. **policy gradient variance**: REINFORCE の sample variance が大きく、
   lr=1e-4 でも recent 勝率 0.17-0.25 で **振動が止まらない**。 lr=5e-5
   まで下げても同じ振動範囲、 lr が原因ではない
2. **value baseline の歪み**: hard matchup (= 勝率 5-15%) で tanh(V(s)) が
   -1 付近で saturate、 advantage = reward - V(s) ≈ 0 となり policy
   update が消える
3. **features 共通の ensemble は seed diversity が効かない**: 2 試行
   (EXT1+EXT3、 EXT3+seed31415) で 共に -2pp 中庸化。 features_v60 の
   表現力が seed-level diversity を吸収する仮説
4. **deck と policy の coupling**: V6 deck で 5000ep warm-start しても
   13.9% に収束 (= fresh と同じ)、 policy weight が我々 deck の strategy を
   焼き込んでいて切替えが効かない

**Submission failure 経験**:
- 53810836 (torch、 single-shot path) → ERROR
- 53812115 (numpy、 single-shot path) → ERROR
- 53812882 (numpy、 multi-root fallback) → COMPLETE 523.1

3 回目で初の COMPLETE。 真因は **`deck.csv` の path 解決が single-shot**
だった = `Path.cwd()` が Kaggle ランタイムで想定外の場所、 3-MLP main.py の
`"deck.csv"` → `"/kaggle_simulations/agent/deck.csv"` の **二段階 fallback**
パターンを継承し損ねた。

**転用可能な insight**:
1. features に新 column を加える時は features の **discriminative power**
   ではなく **learnability** を見る指標が必要 (= ablation study)
2. **submission build は 3-MLP main.py から逸脱しない**。 deck.csv /
   train/* の path 解決パターンを移植時にそのまま使い回す
3. ensemble の seed diversity 効果は features-dependent。 features 共通の
   policy 間で ロジット平均は中庸化のリスク、 **diverse features** や
   **diverse training objective** (= 別 reward function) が必要

### A.3 PIMC v1-v5 の 5 試行 — heuristic value の天井

**設計**: 不完全情報ゲーム標準の Perfect Information Monte Carlo:
1. 相手の手札 / deck 残りを **uniformly sample**
2. `cg.api.search_begin / step` で完全情報シミュ
3. 1-ply (or multi-ply) 進めて value 評価
4. root option ごとの score → argmax

**5 試行の結果**:

| version | feature | subtotal (3 opp × 20g) |
|---|---|---|
| v1 | 1-ply prize delta heuristic | 8.3% (1-ply baseline) |
| v2 | + field-aware (HP/bench/energy weight) | **10.0%** ← peak |
| v3 | + V60 EXT value head 統合 | 1.7-6.7% (weight sweep) |
| v4 | + opp deck inference (overlap で identify) | 10% (= v2 と同水準) |
| v5 | + multi-ply greedy rollout depth 8 | 5.0% (rollout policy が weak) |

**v3 (NN value head 統合) の失敗の本質**:
- V60 EXT の value head は **selfplay 文脈で訓練**
- PIMC は opp_hand を uniform random sample、 search_step で展開した obs は
  selfplay の局面分布と乖離
- value head は そこで信頼できる signal を出さず、 noise として prize_delta
  heuristic を歪める
- → AlphaZero スタイル PIMC には **PIMC 文脈での value head 専用訓練**
  (self-distillation) が必須

**v5 (multi-ply) の失敗**:
- depth=8 で rollout が greedy (= 常に option 0) のため、 8 ply 進めても
  rollout policy が弱い → final prize_delta が信頼できない
- 真の rollout policy には **学習済み policy or rule-based** が必要

**転用可能な insight**:
1. PIMC heuristic は 5-10% で天井、 1-ply / multi-ply の選択は本質では
   ない。 **value function の質**が全て
2. selfplay 訓練の value head を rollout に流用するのは罠 (= 文脈の違いで
   noise 化)
3. opp deck inference は 7 種の vendored deck から overlap-based に identify
   できるが、 **uniform sampling 同等の効果**しか出ない (= サンプル 1 つの
   仮定では不十分、 distribution 推定が必要)

---

## 付録 B: 不採用となった submission の一覧

| ref | type | result | reason |
|---|---|---|---|
| 53776705 | 2-MLP | ERROR | `__file__` 罠 |
| 53776818 | 2-MLP fix | 613.3 | 古い、 3-MLP に更新 |
| 53810836 | V60 EXT3 torch | ERROR | torch 依存 + single-shot path |
| 53812115 | V60 EXT3 numpy | ERROR | single-shot path |
| 53812882 | V60 EXT3 multi-root | 523.1 | 3-MLP より弱い |

最終的に Kaggle で COMPLETE & competitive な submission は:
- 53778627 (3-MLP ensemble): 679.6
- 53793417 (rule_based Iono): 762.2
- 53794617 (rule_based CrustleDashi): 866-888
- 53794828 (rule_based V6): **921.2-926.5** ← LB best

---

## 付録 C: コンペ参加で得た再利用可能な部品

### C.1 `scripts/bench_v60.py`

任意の V60 .pt を 7 vendored opponent で 評価する CLI。 `--games N`
(= per-side N games = 2N/opp) で精度調整。 ストラテジー部門のレポートに
任意の policy ベンチを添付するのに使える。

### C.2 `scripts/extract_v60_weights.py` + `mlp_policy_v60_numpy.py`

torch .pt → numpy .npz 変換 + pure-numpy 推論。 Kaggle で torch が
ない環境でも MLP 推論を実行できる。 forward は matmul + ReLU + tanh の
手書き、 max abs diff 4.77e-07 (= float32 精度内、 実質一致)。

### C.3 `scripts/check_main_exec.py --strict-cwd`

bundle を sandbox に extract、 cwd を bundle の親に設定 + sys.path に
bundle を入れない状態で main.py を exec。 Kaggle の **想定外 cwd** で
deck.csv が解決できない bug を **submit 前** に検出する。

### C.4 `scripts/build_deck.py` + `scripts/build_and_eval_deck.py`

card DB (cards.json) から 60 枚 deck を heuristic で構築。 v1 (= 単一 chain)
から v9 (= ex/non-ex hybrid) まで。 GA loop (`scripts/ga_deck.py`) で
fitness-driven 進化も可能、 ただし noise floor の影響大。

### C.5 `data/matchups.json` + `data/cards.json`

8 系統 rule-based + 我々の 3-MLP/V60 の 7 opp × 80g bench 結果を JSON 化。
新しい agent の fitness を相対評価する material。

---

最後に: 30 サイクル × 30 分 ≈ 15 時間という有限時間予算で、 **どこで
切り上げるか** の判断が学習試行錯誤の核心だった。 ストラテジー部門
report では「成功事例」 だけでなく **「いつ手を引いた / 引くべきだった」
の意思決定ログ** を report の主要価値として提示する。
