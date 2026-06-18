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
