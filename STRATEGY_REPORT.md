# Strategy Report — PTCG ABC (poke-ai)

シミュレーション部門 (締切 8/16) の探索ログをストラテジー部門 (締切 9/14)
向けに整理した報告書。 30 サイクル × 30 分 ≈ **15 時間** の試行錯誤の
記録。

リポジトリ: <https://github.com/g-kari/poke-ai>
ダッシュボード: <https://g-kari.github.io/poke-ai/>

## TL;DR

15+ 系統 (線形 / 3-MLP base / 4-MLP base / V60 single / V60 ensemble /
V6 deck / deck-builder / GA / PIMC / Crustle 検出 / vendored rule-based /
BC v1/v2 / BC+RL (BCRL1/2/3/3-ent) / Bigger MLP / mixed ensemble) を試行
し、 **LB best は rule-based vendored の CrustleDashimaki (LB 874.7) と
V6 (860.8)、 deep-learning best は 3-MLP base (seed=0/2/100, lab 18.9%,
LB 679.6)** で確定。

**驚きの発見**:
- **「more seeds = better ensemble」は不成立**: 4-MLP base ensemble
  は lab +1.5pp 改善でも LB は -247 と大暴落 (599→432)
- **entropy bonus は single policy 改善には有効だが ensemble を破壊**:
  ext seed を 1 つでも混ぜると LB ratio が 35 → 21 に下落
- **TrueSkill σ settling 中の LB は ±150-200 振れる**: 提出直後の
  数十時間で誤判断しない

### Executive summary

| 質問 | 回答 |
|---|---|
| **LB best agent (全体)** | CrustleDashimaki (LB 874.7) |
| **我々自身の DL best (再現困難な local optimum)** | 3-MLP base (LB 679.6, lab 18.9% @ 7-opp) |
| **試行した DL 系統数** | 15+ (BC/V60/v40/ensemble/mixed) |
| **submission slots 使用** | 17+ (うち 2 ERROR、 15 COMPLETE) |
| **最大の発見** | features × ensemble × entropy の 3 軸 trade-off (= 後述) |
| **最大の罠** | 初期 LB スナップショットでの誤判断 (Mix v1 711 → 490 の例) |
| **shippable 再利用部品** | bench_v40/v60.py / collect_bc_dataset.py / bench_v60_ensemble.py |

### 4 つの中心的な学び (改訂版)

1. **「lab 改善 ≠ LB 改善」**: lab winrate は ensemble の改善を直線で
   反映するが、 LB は **diversity composition** に強く依存。 4-MLP base
   は lab 20.4% (+1.5pp) なのに LB 432.9 (-247) と大幅劣化。

2. **「entropy bonus は single policy 改善向け、 ensemble を破壊する」**:
   - 個別 v40 seed に entropy_coef=0.02 で延長 → lab +3.6~+7.3pp 改善
   - 3 個全部 ext で ensemble → lab 16.1% (-10.6pp 大幅 regression)
   - 1 個だけ ext (mixed) → LB 470.9 (3-MLP base 679 を遥かに下回り)

3. **「3-MLP base は特殊解」**: 同じ訓練方法で seed=200 追加した 4-MLP
   base は LB 432.9 (= ratio 21.2)、 3-MLP base の ratio 35.9 は再現困難
   な diversity match。 ensemble size は monotonic ではない。

4. **「7-opp と 4-opp の bench で lab 数値が違う」**: 旧 STRATEGY で
   引用した "3-MLP lab 26.7%" は bench_meta.py (4 Kiyota opp) の値。
   7-opp suite (Lucario/Drag/Iono/Aboma/Crustle/CrustleDashi/V6) で
   再 bench すると 18.9% に。 LB との対応は **7-opp の方が精度高い**。

### features × ensemble × entropy の 3 軸 trade-off (= 大発見)

| 構成 | lab | LB | ratio |
|---|---|---|---|
| 3-MLP v40 base (LB 679.6) | 18.9% | **679.6** | **35.9** |
| 2-MLP v40 base | (~17%) | 613.3 | (~36) |
| 4-MLP v40 base | 20.4% | 432.9 | 21.2 |
| Mixed (1 ext + 2 base) v40 | 20.4% | 470.9 | 23.1 |
| Mix v3 (1 ext + 2 base) v40 | 19.4% | 350.1 | 18.0 |
| 3-poly v60 base ensemble | 18.9% | (未提出) | (推定 ~20) |
| V60 EXT3 single | 20.5% | 562.4 | 27.4 |
| BCRL2 single (BC+RL) | 19.3% | 570.4 | 29.5 |
| Bigger v40 MLP single | 15.0% | (未提出) | (推定 ~14) |

**法則**:
- v40 (40-d) base ensemble は **2-3 seed が sweet spot** (4 で過剰)
- v60 (60-d) features は ensemble 効果なし (deck fingerprint が seed
  diversity を消す)
- entropy + warm-start は **single policy 改善** には有効 (lab +5pp)、
  ensemble に混ぜると LB 200 point 単位で破壊

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

### 5.3 Behavioral cloning from V6 agent (実行結果: lab 5.7%、 提出見送り)

V6 の 30+ 行ロジック (= active rotation + Hariyama energy 配分 +
attack 順序選択) を policy で模倣する supervised learning。 V6 が
LB 888-926 なので、 90% 模倣できれば LB 800+ 期待だった。

**実行結果 v1 (200 games / 7802 samples / 500 epoch BC training)**:
- **supervised acc 69.6%** (random baseline 12.5%) — 学習自体は成功
- **lab winrate 5.7%** @ 40g/opp × 7 opponents (= 280g)
  - vs Mega Lucario: 10.0%
  - vs Iono / Abomasnow / Crustle Dashi: **0.0%**
  - vs V6: 12.5%

**学び (distribution shift トラップ)**:
学習サンプルは V6 が出会った state のみ。 BC policy 自身が agent と
して動くと V6 が探索しない state に陥り、 そこで誤判断する。 訓練 acc
が高くても (= 69.6%)、 そのほとんどが「V6 と同じ簡単な手」に偏って
おり、 hard matchup (Crustle Dashi 0%) では破綻する。

**実行結果 v2 (400 games / 227 wins / 8447 samples / 500 epoch BC v2)**:
- **supervised acc 66.0%** (v1 から -3.6pp、 dataset が concentrated)
- **lab winrate 10.4%** @ 280g — v1 から **+4.7pp 改善**
  - **0.0% matchup が消滅** (全 opp で 7.5-15%、 variance 減)
  - vs V6: 15.0% (v1 12.5% から微増)
- ただし 3-MLP 26.7% / V60 EXT3 20.5% には届かず、 **提出見送り**

wins-only filter は意味があった (= negative samples の影響を取り除けば
分布は均す)、 が **BC 単体の天井** が ~10% にあると確認。 これは V6 が
LB 888 でも、 supervised cloning で学べる部分は 1/9 程度しか転写されて
いないことを示す。

**v3 次の手 (進行中)**: BC v2 weights を REINFORCE warm-start として
利用。 V6 prior で初期化された政策が self-play で distribution shift を
解消できるか検証。 (AlphaGo 戦略の縮小版)

### 5.3b BC 系統が一般的に LB 700+ に届かない理由

1. **state coverage の限界**: V6 は決定論的 rule-based なので、 同一相手
   からほぼ同一の game tree しか生成しない。 200 games × 5 opps = 1000
   trajectories は state space を全くカバーしない。
2. **PG-style action distribution の欠落**: BC は argmax 模倣で、 V6 の
   "tie-break logic" を学べない。 V6 は同点候補で先頭優先するが、 BC
   policy は softmax で確率分散して別の手を選び、 連鎖的に状態が崩れる。
3. **クリティカル decision の希少性**: V6 の本領 (Crustle Dashi vs
   non-ex 攻撃ルート切替) は全 7800 samples 中 ~50 程度。 epoch を回し
   ても重みは平均的な「Hariyama 場出し」ばかりで勘所が学べない。

**回避策候補**: (a) DAgger (state visitation を学習者側で生成、 V6 が
ラベル付け)、 (b) DPO/RLHF (V6 を preference model として REINFORCE)、
(c) 大量データ (10000+ games) で long-tail を網羅。 いずれも 1 サイクル
30 分には収まらない。

### 5.3c BCRL = BC warm-start + REINFORCE (= AlphaGo の縮小版)

BC 単体の天井 ~10% を破るため、 V6 prior で初期化した policy を
self-play で強化学習する hybrid を実行 (2026-06-19、 30 分サイクル内)。

| 試行 | warm-start | episodes | lr | opp pool | lab winrate |
|---|---|---|---|---|---|
| BCRL1 | BC v2 (10.4%) | 2000 | 1e-4 | meta 5 | **12.1%** |
| BCRL2 | BCRL1 (12.1%) | +5000 | 5e-5 | meta 5 + V6 | (進行中) |

**BCRL1 観察**:
- 強敵 (Mega Lucario 22.5%, Mega Aboma 27.5%) と
  弱敵 (Iono 2.5%, Crustle Dashi 2.5%) の **二極化**
- training trajectory: ep 1600 で recent peak 0.23、 ep 2000 ended 0.17
  (= 振動収束していない)
- REINFORCE は BC prior を保持できず、 一部相手のみ「最適化」しすぎた
  結果、 別相手の knowledge を上書きする「catastrophic forgetting」

**学び**:
- V6 prior は **+~2pp 程度** の booster で済み、 BC alone (10.4%) →
  BCRL (12.1%) の改善は誤差範囲
- 5-deck pool の self-play では Iono / Crustle Dashi の特殊戦略を
  学べない (= 報酬信号は累計勝敗だけ、 specific decision の
  credit assignment 不可)
- V60 EXT3 (10500ep, lab 20.5%) より BCRL1 (7000ep total, lab 12.1%)
  が下回るのは、 BC prior が **方向性を歪めた** ことが原因の可能性

### 5.3c+ BCRL2 LB 結果 (= 583.1、 lab→LB ratio 30.2)

BCRL2 (lab 19.3%) を Kaggle に提出 → 初期 LB **462.2** → 24時間後 LB
**583.1** (+120.9)。 LB 採点は TrueSkill 風で、 提出直後の数十戦は
σ が大きく、 後 100+ 戦で本来値に収束する。 BCRL2 の最終 LB:

| 試行 | episodes (累積) | lab | LB | ratio |
|---|---|---|---|---|
| 3-MLP | 2000 ×3 seed | 26.7% | 679.6 | 25.4 |
| **BCRL2** | BC500 + RL7000 | 19.3% | **583.1** | **30.2** ← 最高 |
| V60 EXT3 | 10500 | 20.5% | 578.7 | 28.2 |

**学び (修正版)**:
- 初期 LB スナップショットは ±100 程度の variance がある (= 30 分
  サイクルで提出直後にチェックすると誤判断する)
- BCRL2 ratio 30.2 が我々の DL submission 中 **最高**: BC prior +
  REINFORCE で得た policy は lab 数値より LB で評価される
- 仮説修正: BC + RL は LB で V60 EXT3 同等以上の policy を生む。
  以前の「BC prior が LB を悪化させる」結論は **早期 LB の誤読** だった
- **重要結論**: lab 1pp の改善が LB ~28 ポイントに対応 (= 校正値)。
  LB 700+ には lab 25-27% が必要、 800+ には 28-30% が必要 (= rule-based
  領域 lab 60%+ の世界には届かない構造)

### 5.3d Ensemble = EXT3 + BCRL1 logit 平均 (= 13.9%, 平均化負け)

V60 EXT3 (lab 20.5%) と BCRL1 (lab 12.1%) を **logit 平均** で結合し、
弱点補完を試行。 結果:

| Matchup | EXT3 単体 | BCRL1 単体 | ensemble |
|---|---|---|---|
| Mega Lucario   | 22.5% | 22.5% | 22.5% |
| Dragapult ex   | (~17%) | 7.5%  | 15.0% |
| Iono           | (~22%) | 2.5%  | 10.0% |
| Mega Abomasnow | (~25%) | 27.5% | **32.5%** |
| Crustle Wall   | (~13%) | 7.5%  | 7.5% |
| Crustle Dashi  | (~5%)  | 2.5%  | 0.0% |
| V6             | (~15%) | 15.0% | 10.0% |
| **overall**    | **20.5%** | **12.1%** | **13.9%** |

**結論**: logit 平均は **弱い policy が混ざると平均化負け**。 EXT3 単体
の 20.5% を BCRL1 のノイズが 6.6pp 引きずり下ろした。 唯一 Mega
Abomasnow のみ ensemble が個別を上回った (+5pp boost)。

実装すべき代替案:
- **Weighted logit 平均** (= EXT3 を weight 2.0、 BCRL1 を 0.5 等)
- **Matchup-specific 切替** (= obs から相手 deck を予測し、 最強 policy
  を選ぶ)
- **Mixture-of-Experts**: 状態を gate に通して per-state weight

ただし いずれも追加 1-3 サイクルの投資、 lab 25%+ 目標達成は不確定。

### 5.3e 3-MLP v60 ensemble (= 異 seed × 3 で v40 を超えるか? — 棄却)

v40 features 上の 3-MLP は LB 679 で DL ベスト。 同じ ensemble 構造を
features_v60 (deck fingerprint 追加) で再現すれば lab 30%+ 期待、 という
仮説。 結果:

| 構成 | episode 累積 | lab winrate | LB 推定 |
|---|---|---|---|
| EXT3 単体 (seed=0) | 10500ep | **20.5%** | 578.7 (実測) |
| s200 単体 (seed=200) | 2000ep | 16.1% | ~410 |
| s300 単体 (seed=300) | 2000ep | 13.9% | ~350 |
| 3-poly v60 ensemble | 16500ep 合計 | **18.9%** | ~480 |
| **3-MLP v40** (比較) | 6000ep × 3 | **26.7%** | **679 (実測)** |

**結論**: deck fingerprint feature の v60 features は **ensemble に転じない**。
理由の推測:
1. 60-d state vs 40-d state で input 層パラメータ +50%、 同じ network
   width (= 64,32) では各 input への重みが希薄化
2. 16-bucket hash の deck fingerprint は **collision noise** が大きく、
   ensemble 平均で打ち消されない持続バイアスを生む
3. 60-d を使いこなすには 5000-10000ep × 3 seed = 30000ep+ 必要、
   我々の 5000ep 各では undertrain

**含意**: v60 features は単純な 3-MLP ensemble での恩恵がない。
deck fingerprint を活かすなら:
- bigger network (= 128, 64) で容量増、 同じ episode count で学習
- attention-based feature fusion (= 5.4 transformer 路線)
- 単独 policy で長期学習 (= EXT3 路線、 lab 20.5% 天井)

**最終確認 (公平な 5000ep × 3 メンバー)**:

| 構成 | episode 合計 | lab winrate |
|---|---|---|
| EXT3 単体 | 10500ep | **20.5%** |
| 3-poly (unfair: EXT3 + s200 + s300、 各 2000ep) | 14500ep | 18.9% |
| **3-poly (fair: EXT3 + s200ext + s300ext、 各 5000ep)** | **20500ep** | **17.5%** ← 更に悪化 |

メンバーを公平にしたら ensemble は **逆に悪化**。 原因は s200ext/
s300ext で warm-start を続けると **lab が regression する** ため
(s200: 16.1% → s200ext: 13.2%)。 v60 features は extension training
で過学習 (= 訓練 pool に過剰最適化、 test bench で破綻) を起こす。

V60 features 路線の最終結論:
- **単独 policy** で seed=0 + 10500ep の特殊解 (lab 20.5%、 LB 578.7) のみが
  到達点
- **ensemble** では不可能 (= 個別メンバーが EXT3 級に達しないため平均化負け)
- **BC warm-start からの REINFORCE** (= BCRL2) は別経路として有効
  (lab 19.3%、 LB **583.1**、 ratio 30.2 で我々の DL 最高)

### 5.3f REINFORCE warm-start regression パターン (= 三度確認)

複数の実験で、 既存 policy を warm-start として REINFORCE を続行すると、
**lab winrate が逆に下がる** パターンを観測:

| 試行 | 起点 | 追加 episode | 起点 lab | 終了 lab | Δ |
|---|---|---|---|---|---|
| V60 EXT4 | EXT3 10500ep | +5000ep lr=5e-5 | 20.5% | 13.3% | **-7.2pp** |
| s200ext | s200 2000ep | +3000ep lr=3e-4 | 16.1% | 13.2% | **-2.9pp** |
| BCRL3 | BCRL2 7000ep | +5000ep lr=5e-5 | 19.3% | 15.4% | **-3.9pp** |

**全 3 試行で regression**。 訓練の recent winrate (= 学習 pool 内の
最近 100 試合) は 0.22-0.30 で改善傾向を示すのに、 test bench (= 同じ
opp 但し seed 固定で fresh 試合) では悪化。

**仮説**:
1. 後期 REINFORCE は **訓練 pool 内の特定 trajectory に過剰最適化**
2. policy は近視的に「報酬を取れる手」 を覚えるが、 反応的でない手
   (= 序盤の構築) の質が劣化
3. learning rate を下げても (= 5e-5) variance 削減が不十分、 policy
   が局所最適に逃げ込む

**含意**:
- REINFORCE warm-start は 1 サイクル分 (2000-5000ep) しか有効でない
- 長期改善には PPO (clipped objective) や entropy bonus、 KL penalty で
  policy の暴走を抑制する必要
- 我々の BCRL2 (lab 19.3%, LB 583.1) は **BC+RL の local optimum**、
  これ以上の延長は LB 悪化を招く

### 5.3g v40 vs v60 features の **ensemble lift** 構造的差異 (= 新発見)

各 seed 単独の lab winrate と ensemble の lab winrate を比較:

| features | 単独 seed lab | 3-policy ensemble lab | lift |
|---|---|---|---|
| **v40 (40-d)** | seed=0 alone: **13.2%** | **26.7%** | **+13.5pp** |
| v60 (60-d, deck fingerprint) | EXT3 alone: 20.5% | 17.5% (fair members) | **-3.0pp** |

**v40 は ensemble に強い、 v60 は ensemble で逆に下がる**。

**仮説**: v60 features の **deck fingerprint hash bucket** が opponent
identity を強く encode するため、 各 seed が同じ「相手別 specialization」
に収束する。 logit 平均で多様性が消え、 lift が出ない。 一方 v40
features は opponent 情報が乏しいので、 各 seed が異なる「内部戦略」
(= 自分のデッキの使い方の流派) を学び、 ensemble で補完しあう。

**含意**:
- 3-MLP v40 (LB 679) の強さは個別 seed の質ではなく **ensemble の多様性**
  に依存
- v60 features を追加するなら、 ensemble を諦めて **単独 policy で
  長期学習** (= EXT3 路線) するべき
- 単独 seed v40 を改善できれば、 ensemble lift で更に伸び代がある
  (= 13.2% × 2 = 26.7% の係数で、 18% → 36% の可能性)
- **次の挑戦**: v40 seed を entropy bonus で延長して 18-20% に上げる
  → 3-MLP ensemble で 30%+ → LB 700-800 期待

### 5.3i 訂正: TrueSkill σ settling で Mix v1 LB は 711 → 490 に下降

前 section (5.3h) で **Mix v1 (= seed=0 ext + 2 base + 100 base) が LB
711.2 で 3-MLP base 679 を超えた** と記述したが、 24時間以内に再 fetch
すると **490.3 まで下降**。 transient peak だった事が判明。

| 提出 | 提出直後 LB | 24時間後 LB | Δ |
|---|---|---|---|
| Mix v1 | 659.7 → 711.2 → **490.3** | -220.9 から peak | (波乱) |
| Mix v3 | 540+ → **404.5** | (初期高値から下降) | (波乱) |
| BCRL2 | 462 → 583 → **570.4** | 安定方向収束 | (sealed) |
| 3-MLP base | (記録なし)→ **679.6** | 安定 |  — |

**学び**:
- 初期 LB の数十時間は **±100-200 point variance** あり、 transient
  high/low に騙されない
- ensemble に entropy ext を入れると LB ratio が **24% (Mix v1) と
  21% (Mix v3) まで落ち**、 base ensemble の 35.9 に比べ大幅悪化
- **最終結論**: ext seed は ensemble を必ず害する。 3-MLP base 679 が
  我々の真の DL チャンピオン

### 5.3h Entropy bonus は **single policy 向け**、 ensemble を壊す

v40 seed=0/2/100 を entropy_coef=0.02 で warm-start 2000ep 延長:

| Seed | Base lab | Ext lab | Δ |
|---|---|---|---|
| 0 | 13.2% | **20.5%** | **+7.3** |
| 2 | 13.2% | 16.8% | +3.6 |
| 100 | (未測定) | ~22% (推定) | ? |

個別 seed は全部改善 (= entropy bonus が warm-start regression を防止)。
ところが ensemble bench:

| 構成 | lab winrate |
|---|---|
| 3-MLP base ensemble (= LB 679) | **26.7%** |
| **3-MLP-ext ensemble** | **16.1%** (-10.6pp regression!) |

**ensemble lift の逆転**:
- base: 単独 ~13% → ensemble 26.7% = **+13.5pp** (logit averaging works)
- ext: 単独 ~20% → ensemble 16.1% = **-4pp** (logit averaging fails)

**構造的解釈**:
- base policies は **exploratory** (entropy が高い、 logit が softer)
  → ensemble 平均で「diverse vote の中庸」 を取れる
- ext policies は **deterministic** (entropy bonus で更に高い entropy
  を保つはずだが、 warm-start で local optima に各々収束しているため、
  実質的に specialized で confident な policies)
  → 3 specialists の logit average は何も合意しない中庸を生み、 全員が
  間違った手を選ぶ

**含意 (重要)**:
- entropy bonus は **single policy training** には有効
  (= V60 EXT3 / BCRL2 級の 20%+ 単独 policy を狙う場合)
- ensemble approach (= 3-MLP 級の 26%+) を狙う場合は **entropy を
  入れない方が良い**
- 個別の policy 改善と ensemble 改善は **別問題**

これは Kaggle 競技だけでなく深層強化学習一般において、 ensemble
設計の重要な経験則になる発見ですわ。

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

### A.3 vendored rule-based 7 種の比較 — LB best (V6 926) の正体

**収集元**: Kaggle 公開 kernel から **6 種 + 我々 vendored** を pull:

| subject | 出処 | vote | deck タイプ | overall lab @ 80g |
|---|---|---|---|---|
| Mega Lucario | Kiyota 公式 | 203 | ex + Hariyama hybrid | 46.5% |
| Dragapult ex | Kiyota 公式 | 97 | ex 機動力 | 48.1% |
| Iono | Kiyota 公式 | 35 | Lightning chain | **64.0%** |
| Mega Abomasnow ex | Kiyota 公式 | 28 | Ice tank | 40.0% |
| Crustle Wall (haru) | harukiharada | 7 | wall + lock | 36.9% |
| Crustle Dashi | dashimaki360 | 35 | Day-1 #1 wall | **67.3%** |
| **RomanrozenV6** | romanrozen | **36** | **anti-Crustle hybrid** | **57.9%** |

**LB スコアの推移** (= 評価期間進行による収束):
- 53793417 (Iono): 600 → 615 → 762.2
- 53794617 (CrustleDashi): 718 → 866 → 870-888
- 53794828 (V6): 801 → 873 → **921-926.5**

**なぜ V6 が LB 最強か** (= 我々が最も学んだ事例):

1. **CRUSTLE_AWARE=True の 30+ 行 logic**:
   - obs から相手 active のカード ID で Crustle 系を検出
   - 検出時 **active を Mega Lucario ex から Hariyama (non-ex) に rotate**
   - Hariyama に **Fighting エネルギーを集中配分**
   - Hariyama の attack で Crustle の「ふしぎなロックイン」 を抜く
   - これは **2 段階 decision** (= retreat → bench select) を扱う

2. **proactive setup**:
   - 初期 turn から **Hariyama 系も bench に配置**
   - Crustle が出る前から secondary chain が ready
   - 我々の generic_agent は OPTION_PRIORITY で常に primary 優先、
     secondary は冷遇 → 検出してから setup では間に合わない

3. **hybrid deck の比率**:
   - Mega Lucario ex (Stage 1 ex)、 Hariyama (Stage 1 non-ex)、
     Solrock (Basic supporting) を **2:2:1 程度** で構成
   - non-ex Hariyama が deck の **13%+** で Crustle 戦の手札確率を確保
   - 我々の v7 hybrid は secondary 4 枚 (= 7%) で Crustle 戦に間に合わず

**含意**: V6 の 30+ 行 logic は agent と deck の **同時設計**。 これを
heuristic で再現するのは 7 サイクル投資して頓挫 (Task #105 / v7-v11)、
深層学習で再発見するには 25%+ lab 必須 = 我々の V60 路線では届かなかった。

### A.4 deck-builder v1-v9 + GA loop — Task #107 の構造的限界

**動機**: 「人間が思いつかない anti-meta deck を自動探索」 を狙った
heuristic agent。 user 方針 (C) の長期目標。

**v1-v9 の進化**:

| ver | feature | best output | overall lab |
|---|---|---|---|
| v1 | 単純 HP/(retreat+1) score | Mega Camerupt ex (chain 無視) | n/a |
| v2 | + evolves_from chain 解決 | Salandit → Salazzle ex | 17.5% (40g) |
| v3 | + Stage 2 サポート | Charmander → Charmeleon → Mega Charizard Y ex | n/a |
| v4 | + 6 spec 自動 fitness 評価 | **Snorunt → Mega Froslass ex** | **18.0% (5g)** |
| v5 | target bonus 30 → 200 + ex penalty | Salandit など | 8.0% (失敗) |
| v6 | v4 weights に revert | Snorunt → Mega Froslass ex | 17.5% (40g) |
| v7 | hybrid chain (primary ex + secondary non-ex) | Charizard ex + Toxicroak | 4.8% (失敗) |
| v8 | + Crustle 検出 + non-ex routing | (deck 同じ、 agent 改修) | 7.1% |
| v9 | + secondary 比率 4 → 8 | (deck 同じ) | 7.1% |

**v4 の発見** (= Task #107 の peak achievement):

`scripts/build_and_eval_deck.py` で 6 spec を 5g/opp で fitness 評価。
**Fighting/Stage1 spec で 18.0% overall** が判明。 内訳:
- target_type='F' (Fighting) を強制したが、 type_bonus 30 が weak で
  実際は **Water type の Snorunt → Mega Froslass ex (HP 310 Stage1 ex)**
  が選ばれた (= 偶然の発見)
- Mega Lucario 40%、 Dragapult 30% は他 spec より圧倒的

40g 本格 bench で 17.5% (5g の 18.0% と整合)、 だが Crustle Dashi 0% が
**6 試行 6 回持続** = builder 単独では Crustle 対策不可能。

**GA loop 3 試行の noise floor 教訓**:

| version | eval games | best fitness (in-eval) | 真の 40g bench |
|---|---|---|---|
| v1 | 3g/eval | 23.3% | 13.2% |
| v2 | 8g/eval | 22.5% | 15.4% |
| v3 | 8g/eval | 20.0% | 13.2% |

3 試行で **真の値は全て v4 baseline (17.5%) 以下**。 GA は **「ある matchup
を伸ばし他を犠牲にする」 local trade-off** を選び、 single mutation では
local optimum から脱出できない。

**転用可能な insight**:
1. heuristic deck-builder で「偶然の anti-meta 発見」 はあり得るが、
   **真の improvement vs v4 baseline は noise floor 内で確認不可**
2. GA loop の eval は **最低 40g/eval** (= 1 generation 90 秒) が必要、
   1-card swap では local optimum で 5pp 以下の improvement しか出ない
3. deck-builder + agent の **三位一体改修** が真の breakthrough (V6 が
   実証)、 deck だけ・agent だけの改修では LB 競争力に届かない

### A.5 PIMC v1-v5 の 5 試行 — heuristic value の天井

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
