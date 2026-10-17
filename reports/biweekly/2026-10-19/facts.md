# 2026-10-19 進捗報告 ファクトシート（plan 0103 Phase 1〜現在、plan 0107）

作成 2026-10-16 00:50 JST 時点。数値はすべて出典の値をそのまま写した。
「派生」と書いた数値だけが、出典の2つの値の単純な差（こちらで計算）。出典が見つからない値は「未確認」。

出典の略記:
- `P0103:Lnnn` = `docs/plans/0103-lightweight-imagenet-architecture-survey.md` の行
- `REG:Lnnn` = `docs/plans/0103-discussion-register.md` の行
- `D00nn:Lnn` = `docs/decisions/00nn-*.md` の行
- `P0107:Lnn` = `docs/plans/0107-train-time-capacity-deploy-plain.md` の行
- `DBG30:Lnn` = `docs/debugging/0030-convstem-dc-offset-pgd-at-collapse.md` の行
- `P0104:Lnnn` = `docs/plans/0104-finetune-lr-schedule.md` の行
- `TRV:Lnn` = `tests/unit/test_registry_variants.py` の行（パラメータ数・MAC の固定値）
- `run:<run id>`（H）= Hamster の `/home/shunsukenaito/workspace-local/ard-runtime/ard_codex_bootstrap/runs/<run id>/outputs/train/epoch-metrics.jsonl` の最終行
- `run:<run id>`（F）= Ferret の同じパスの最終行（2026-10-16 00:48 JST に ssh で読んだ）
- run ファイルの値は小数（fraction）を ×100 して小数第2位で表示した。Phase 1 の行は `val_clean_accuracy` / `val_pgd_accuracy`（取り分けた 25,620 枚すべて）、Phase 2 の行は `val_full_*`（同じ 25,620 枚）。

注意（全体）: 数値はすべて内部検証（学習データから取り分けた 25,620 枚）で、公式テストではない。AutoAttack も公式評価もまだ（例 D0042:L29）。seed は特記のない限り 0 の1本。

---

## 1. 共通設定

| 項目 | 値 | 出典 |
|---|---|---|
| データ | ImageNet-1k（元画像）、224 px（全モデル） | D0042:L26, REG:L26 (B5) |
| 初期化 | スクラッチ（ランダム初期化）のみ。事前学習は使わない（2026-10-01 決定） | REG:L117 |
| 手法 | PGD-AT、教師なし | D0042:L26 |
| 学習時の攻撃 | PGD-3、L∞ ε=4/255、step 8/765、ランダムスタート、評価モードで生成 | REG:L29 (B8), D0042:L27 |
| 評価・選択の攻撃 | PGD-10（同じ ε と step） | D0042:L27 |
| バッチ / GPU | 128、world size 1 | D0042:L28, REG:L106–109 (B10) |
| epoch | 50（full AT）。30 epoch は −1.7pt で不採用 | REG:L120 |
| スケジュール | 10 epoch ウォームアップ、epoch 25 と 38 で 1/10（multistep） | D0042:L28 |
| データ拡張 | ランダムな切り抜き＋左右反転のみ | REG:L28 (B7) |
| 評価集合 | 学習データから取り分けた 25,620 枚（内部検証）。公式 val は約3pt 低く出る | D0042:L29, REG:L38 (C3) |
| Phase 2 の評価 | 毎 epoch は固定 5,000 枚サブセット、学習の最後に best と last を 25,620 枚すべてで評価。EMA 0.9999 を並行保持 | REG:L122, D0042:L23 |
| 判定（Phase 2） | PGD-10 の差が ≥1.5pt「効果あり」、0.5〜1.5pt「保留」（seed 2本目で確認）、<0.5pt「効果なし」。最後の epoch・普通の重みで読む | REG:L121, D0042:L48 |
| 2本の差の SE | clean 約 0.43pt、PGD-10 約 0.37pt | D0047:L40, D0050:L60 |
| seed 0 と 1 の差 | MobileNetV4-S 0.36、MobileNetV4-M 0.09、ConvNeXt-Atto 0.36pt | REG:L129 |
| 効率の物差し | Hamster RTX 4090 での計算時間＋FLOPs | REG:L119, REG:L137 |

### 最適化手法の違い

| 系統 | 最適化 | 設定 | 出典 |
|---|---|---|---|
| BN の CNN（MobileNetV4-S/M、EfficientNet-B0） | SGD | Nesterov、momentum 0.9、wd 1e-4、ピーク LR 0.025 | D0042:L28, REG:L101 |
| LN のモデル（ConvNeXt-Atto、DeiT-Tiny） | AdamW | betas 0.9/0.999、wd 0.05、ピーク LR 1e-3 | D0034:L26, D0052:L32, REG:L129（DeiT 1e-3 確定） |
| 経緯 | 系統ごとの基準レシピ（BN→SGD、LN→AdamW）は人間の決定 2026-09-28。SGD の DeiT-Tiny run は途中停止（結果として扱わない） | | REG:L101 |
| SGD での ConvNeXt-Atto（参考、崩れ） | 30.68 / 16.53 | | run:plan0103-phase1-convnext-atto-random-v1 (H), D0048:L41 |

### 実行モードの違い

- Phase 1 は非決定的＋cuDNN 自動選択。MobileNetV4-S の行だけ決定的モード（＋CUDA Graphs）（REG:L30 B9, P0103:L3230）。
- Phase 2 の MobileNetV4-S は決定的（＋CUDA Graphs、SiLU/TRADES/SE 等は eager）。ConvNeXt-Atto・DeiT-Tiny・MobileNetV4-M・EfficientNet-B0 は非決定的（REG:L128–129）。
- plan 0107 以降は非決定的が基本（REG:L138）。

### モデル（パラメータ数と MAC）

| モデル | params | MACs | 出典 |
|---|---:|---:|---|
| MobileNetV4-Conv-S | 3,774,024 | 186,005,568 | TRV:L32（概数 3.77M / 0.19 GMACs: P0103:L365） |
| MobileNetV4-Conv-M | 9,715,512 | 827,649,984 | TRV:L33（9.72M / 0.83: P0103:L367） |
| EfficientNet-B0 | 5.29M | 0.39 G | P0103:L366（TRV に正確値なし） |
| ConvNeXt-Atto (V1) | 3,695,520 | 547,332,480 | TRV:L35（3.70M / 0.55: P0103:L368） |
| DeiT-Tiny | 5,717,416 | 1,074,851,328（＋attention 178,831,872） | TRV:L34（5.72M / 1.07: P0103:L369） |

Phase 2 の派生モデル: SE 版 3,772,296 / 185,999,424（TRV:L44）、SE fullhead 4,023,432（TRV:L46）、deep-narrow 2,758,636 / 554,452,128（TRV:L49）、ols 3,702,912 / 576,534,912（TRV:L50）、ConvNeXt ConvStem 3,701,400 / 570,664,320（TRV:L51）、DeiT ConvStem 5,826,280（TRV:L48）。SiLU・hard-swish 版はベースと同じ（TRV:L41,L43,L47）。

### 学習速度（Hamster RTX 4090、224 px、バッチ 128、PGD-3）

2026-10-11 の測定（40バッチ×2エポック）。単位 img/s。左から「非決定的＋自動選択 / 非決定的＋CUDA Graphs / 決定的＋CUDA Graphs」（REG:L129）:

| モデル | img/s | 出典 |
|---|---|---|
| MobileNetV4-S | 1364 / 1399 / 1145 | REG:L129 |
| MobileNetV4-M | 522 / 512 / 435 | REG:L129 |
| EfficientNet-B0 | 346（compile なし）/ 340 / 288 | REG:L129 |
| ConvNeXt-Atto | 741（非決定的）/ 628（非決定的＋CG）/ 611（決定的＋CG） | REG:L128 |
| DeiT-Tiny | ベンチマーク値は未確認。実 run で 557〜559 img/s（lr 5e-4、Hamster GPU0、隣の GPU 使用中） | D0034:L37 |

参考: バッチを上げても1枚あたりは速くならない（MobileNetV4-S 128 で 1685、1024 で 1465 img/s）（REG:L109）。

---

## 2. Phase 1（各モデル LR 3点以上、full AT 50 epoch、seed 0 で LR を選択）

選択規則: 最後の epoch の内部検証 PGD-10 で LR を選ぶ（P0103:L3251）。学習率は全モデル3点ずつ探索し最良同士で比べる（REG:L118）。

| モデル | 最適化 | LR grid → 最後の PGD-10 | 最良 LR | seed 0 clean / PGD-10 | seed 1 clean / PGD-10 | seed 差 (PGD-10) | 時間 / 速度 |
|---|---|---|---|---|---|---|---|
| MobileNetV4-S | SGD | 0.0125 / 0.025 / 0.05 / 0.1 → 29.16 / 30.71 / 30.05 / 29.18（P0103:L3255, P0104:L221） | 0.025 | 53.85 / 30.71（run:plan0103-mnv4s-fullat-50ep-lr0025-cg-s0-v1 (H)、D0042:L37） | 53.63 / 30.35（run:plan0103-mnv4s-fullat-50ep-lr0025-cg-s1-v1 (F)） | 0.36（REG:L129） | 15.45 h（Hamster、CUDA Graphs）（P0103:L3231） |
| MobileNetV4-M | SGD | 0.0125 / 0.025 / 0.05 → 37.00 / 39.39 / 39.10（P0103:L2816） | 0.025 | 63.79 / 39.39（run:plan0103-phase1-mobilenetv4-conv-medium-random-v1 (F)。docs では clean「未読」D0040:L34） | 63.71 / 39.30（D0040:L32） | 0.09（REG:L129） | seed 1: 35.39 h、515〜521 img/s（Hamster GPU0）（D0040:L41） |
| EfficientNet-B0 | SGD | 0.0125 / 0.025 / 0.05 → 34.25 / 35.93 / 35.88（P0103:L3795） | 0.025 | 58.22 / 35.93（run:plan0103-phase1-efficientnet-b0-random-v1 (F)。docs では clean 未確認 D0037:L35） | 58.31 / 35.76（run:plan0103-phase1-efficientnet-b0-random-s1-v1 (F)。docs 未記録） | 0.17（派生） | lr 0.05 セル: 48.62 h、373〜379 img/s（Hamster）（D0036:L41）。lr 0.025 セル（Ferret）の時間は未確認 |
| ConvNeXt-Atto | AdamW | 2.5e-4 / 5e-4 / 1e-3 / 2e-3 → 34.23 / 36.35 / 37.02 / 36.65（P0103:L3797–3798。1.25e-4 は取り消し） | 1e-3 | 59.96 / 37.02（run:plan0103-phase1-convnext-atto-adamw-random-lr1em3-v1 (F)。docs では clean 未記録 D0052:L45） | 59.50 / 36.66（run:plan0103-phase1-convnext-atto-adamw-random-lr1em3-s1-v1 (F)） | 0.36（REG:L129） | Phase 1 セルは Ferret（時間未確認）。同モデル Hamster の Phase 2 cosine: 25.23 h、713〜716 img/s（D0052:L60） |
| DeiT-Tiny | AdamW | 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 → 27.19 / 31.25 / 33.27 / 35.00 / 34.66（P0103:L3800–3801, REG:L129） | 1e-3 | 58.63 / 35.00（run:plan0103-phase1-deit-tiny-adamw-random-lr1em3-v1 (F)） | 57.10 / 33.63（run:plan0103-phase1-deit-tiny-adamw-random-lr1em3-s1-v1 (F)。docs 未記録） | 1.37（派生。docs 未記録、判定基準の見直しは未実施） | lr 5e-4 セル: 33.00 h、557〜559 img/s（Hamster）（D0034:L37）。lr 1e-3 セル（Ferret）の時間は未確認 |

補足:
- 中間値（clean / PGD-10）: ConvNeXt-Atto 2.5e-4 55.98 / 34.23、5e-4 58.83 / 36.35。DeiT-Tiny 2.5e-4 54.54 / 31.24（ep 49）、5e-4 56.85 / 33.27（P0103:L3353–3355）。
- EfficientNet-B0: 0.0125 は 56.08 / 34.25、0.05 は 57.86 / 35.88（D0037:L34,L36）。
- 判定基準の見直しは「EfficientNet-B0 と DeiT の seed 1 がそろってから」（REG:L129）。両方の run は Ferret で完了しているが、docs への取り込みは未確認。

### Phase 1 の主な知見

**(a) 2段階（AdvXL 型）vs full AT** — full AT を採用
- 設計: 112 px・PGD-1 で第1段階 → 224 px・PGD-3 で20 epoch 仕上げ。1段階 50 epoch と FLOPs を揃える（第1段階 195 epoch）。判定ルール ±1pt は事前登録（REG:L105）。
- 事前登録の判定（195 epoch 版）: 54.91 / 28.45、PGD-10 −2.26pt で「悪化」（P0103:L2804）。
- 事後の変種: 第1段階 100 epoch 55.12 / 28.19、50 epoch（S=256）53.13 / 29.84（−0.87）（P0103:L2807–2808）。第1段階が長いほど第2段階後の PGD-10 が悪い（P0103:L2810）。
- 最速設定での時間（MobileNetV4-S、Hamster）: full AT 50 epoch 53.85 / 30.71、15.45 h。2段階 50＋20 epoch 53.13 / 29.84、3.45＋6.30 = 9.75 h。比 0.63（P0103:L3230–3234）。
- 第2段階の長さ: 5 epoch 51.57 / 28.77（不合格）、10 epoch 52.18 / 29.38（合格）（P0103:L3237）。
- 一般化の判定（両モデルとも 1pt 以内が条件）: MobileNetV4-M 63.04 / 37.99 vs 39.39、−1.40 で不合格。EfficientNet-B0 は第2段階が OOM（第1段階のみ 49.97 / 24.74）（P0103:L3241–3244）。→「Phase 1 は full AT、50 epoch」（P0103:L3246）。

**(b) 事前学習をやめた**
- MobileNetV4-S で初期化ごとに LR 3通りの最良同士: 事前学習ありが PGD-10 +1.21pt、clean +3.4pt（seed 1本、ノイズの範囲に近い）（REG:L48 D1）。参考値: 事前学習あり lr 0.015 57.27 / 31.92（run:plan0104-mobilenetv4-pretrained-lr0015-v1 (F)）vs ランダム初期化 53.85 / 30.71。
- 決定（人間、2026-10-01）: 事前学習モデルは使わない（スクラッチのみ）。MobileNetV4-M の事前学習ありの run は途中で止めた（REG:L117）。
- 背景: 事前学習の重みはモデルごとに作り方（epoch 数・拡張・蒸留・解像度）が違う（REG:L25 B4）。

**(c) 30 vs 50 epoch**
- 30 epoch は PGD-10 29.03、基準 30.71 → −1.7pt。50 epoch を採用（REG:L120）。30 epoch run: 51.16 / 29.03（run:plan0103-mnv4s-fullat-30ep-cg-s0-v1 (F)）。
- 50 epoch の頭打ちは LR スケジュールによる。clean 学習の 50→100 epoch の差 G = +0.09pt（REG:L103）。ただし学習画像への当てはまりの水準は大きく違う（clean 学習の clean 約78% vs 敵対的学習の PGD-10 約35%）（REG:L103）。clean 学習の内部検証は 70.86%（50 epoch）（REG:L99）。clean 学習の cosine 対照 71.28 vs multistep 70.86（P0103:L2813）。

**(d) LR への敏感さ**
- MobileNetV4-M で LR を半分にすると PGD-10 −2.4pt（0.0125: 37.00、0.025: 39.39）、揺れの幅を超える → 全モデル3点探索（案A）（REG:L118）。
- AdamW の2モデルは当初の範囲（〜5e-4）の上端が最良だったため上方へ延長（P0103:L3356–3357, P0103:L3456–3459）。ConvNeXt-Atto 2.5e-4→5e-4 で PGD-10 +2.1pt（P0103:L3461–3462）。
- MobileNetV4-S は最良付近がなだらか（0.0125〜0.1 の範囲で 29.16〜30.71）（P0103:L3255, P0104:L221）。
- ImageNet-100 の代理学習は LR 選択には使えない（順位が逆転）。初期化の差の向きには使える（REG:L104）。

---

## 3. Phase 2 の1要因比較

列: Δ = 候補 − 比較相手（最後の epoch・普通の重み・25,620 枚）。判定は出典に記録された文言。

### MobileNetV4-S（比較相手: 基準＋EMA 53.85 / 30.71、D0042:L33）

| 要因 | clean / PGD-10 | Δ clean | Δ PGD-10 | 判定（記録どおり） | decision |
|---|---|---:|---:|---|---|
| 基準＋EMA（設定チェック） | 53.85 / 30.71（Phase 1 とビット単位で一致） | 0.00 | 0.00 | 設定の影響なし | 0042（D0042:L39–40） |
| EMA（EMA の重み） | 53.95 / 30.80 | +0.10 | +0.10 | 効果なし | 0042（D0042:L34,L45） |
| cosine | 55.09 / 31.49 | +1.24 | +0.78 | 保留 | 0038（D0038:L37,L42,L46） |
| ラベル平滑化 0.1 | 53.35 / 31.16 | −0.50 | +0.45 | 効果なし（境界の 0.05pt 下） | 0039（D0039:L38,L44,L48） |
| mixed batch（Kurakin、λ=0.3、BN 共有） | 62.88 / 17.34 | +9.03 | −13.37 | 1.5pt の約9倍の悪化 | 0041（D0041:L39,L47,L49） |
| mixed batch（BN 分割） | 47.95 / 22.46 | −5.90 | −8.25 | 効果なし（8.25pt の損）。BN 共有版比 +5.12 | 0049（D0049:L36,L43–45）。線は閉じた（REG:L131） |
| TRADES β=6 | 54.49 / 26.06 | +0.64 | −4.65 | 「効果あり」の大きさで向きは損 | 0047（D0047:L33,L40–41） |
| AWP（γ=0.01、warmup 0） | 28.47 / 17.45 | −25.38 | −13.26 | 「効果あり」の大きさで向きは損。有効でないと結論 | 0048 chosen A（D0048:L34,L41–42, REG:L133） |
| IDBH 弱 × 50 ep | 52.48 / 29.29 | −1.37 | −1.42 | 「保留」の幅だが向きは悪化 | 0045（D0045:L35,L41,L43） |
| IDBH 弱 × 100 ep（vs 標準×50） | 54.89 / 30.29 | +1.04 | −0.42 | 効果なし | 0046（D0046:L33,L40,L43） |
| IDBH 弱 × 100 ep（vs 標準×100） | 54.89 / 30.29 | −1.32 | −1.59 | 悪化側。IDBH を閉じる | 0054（D0054:L40,L42） |
| 標準拡張 × 100 ep | 56.20 / 31.88 | +2.35 | +1.17 | 保留（計算は2倍） | 0054（D0054:L30,L38,L44） |
| weight decay 除外（ReLU） | 53.47 / 29.57 | −0.38 | −1.14 | 「保留」の帯で向きは下がる。最終的に「保留」 | 0043（D0043:L34,L40,L42）, REG:L134 |
| SiLU | 55.45 / 33.27 | +1.60 | +2.56 | 効果あり | 0044（D0044:L35,L41,L43） |
| SiLU＋wd 除外（vs SiLU） | 56.50 / 33.66 | +1.05 | +0.39 | SiLU では効果なし | 0050 chosen A（D0050:L34,L48,L50） |
| SE（パラメータ数をそろえた版、ReLU） | 54.32 / 31.12 | +0.47 | +0.41 | 効果なし（境界の近く） | 0051 chosen A（D0051:L34,L42–43） |
| SiLU＋SE（vs SiLU） | 56.42 / 33.83 | +0.98 | +0.55 | 保留（境界のすぐ上）。SE は採用しないで閉じた | 0055 chosen A（D0055:L35,L48–51）, REG:L137 |

参考: SiLU＋SE は基準＋EMA 比 +3.12pt で、MobileNetV4-S 50 epoch で最高（D0055:L57）。2×2 表: 活性化×wd（D0050:L42–45）、活性化×SE（D0055:L46–49）、拡張×epoch（D0054:L36–40）。

### ConvNeXt-Atto（比較相手: 基準＋EMA 59.69 / 36.92、D0053:L34）

| 要因 | clean / PGD-10 | Δ clean | Δ PGD-10 | 判定 | decision |
|---|---|---:|---:|---|---|
| 基準＋EMA（設定チェック、vs Phase 1 37.02） | 59.69 / 36.92 | 未確認（Phase 1 clean は docs 未記録。run ファイルでは 59.96） | −0.10 | 設定の影響なし（ビット一致ではない） | 0053（D0053:L42–44） |
| EMA（EMA の重み） | 59.72 / 36.99 | +0.03 | +0.06 | 効果なし | 0053（D0053:L35,L52） |
| cosine | 60.70 / 37.90 | +1.01 | +0.98 | 保留（Phase 1 比でも +0.88 で保留） | 0052 / 0053（D0052:L38,L43,L47, D0053:L45） |
| ラベル平滑化 0.1 | 59.63 / 37.61 | −0.06 | +0.69 | 保留（PGD のみの評価、勾配の隠れは未確認） | 0058（D0058:L35,L41–42,L54） |
| 深く細い版（(3,3,8,1)、幅 ×0.9） | 57.97 / 35.53 | −1.72 | −1.39 | 「保留」の帯だが向きは下がる | 0056（D0056:L33,L40–41） |
| ols（timm の overlapping linear stem） | 59.29 / 36.45（EMA 59.29 / 36.64） | −0.40（派生） | −0.47（派生） | 未記録（Ferret で 50/50 完了、record・decision なし） | なし。run:plan0103-phase2-convnext-atto-ols-s0-v1 (F) |
| ConvStem（β2 0.999） | 0.10 / 0.10（50 epoch ずっと当てずっぽう） | — | — | 学習しなかった（結果なし）。B' で β2 0.95 を再試行 | 0057 chosen B'（D0057:L28–37, REG:L139） |

### MobileNetV4-M

| 要因 | clean / PGD-10 | 比較相手 | Δ clean | Δ PGD-10 | 判定 | decision |
|---|---|---|---:|---:|---|---|
| SiLU | 61.26 / 38.33（EMA 61.37 / 38.69） | Phase 1 ReLU seed 1: 63.71 / 39.30 | −2.44（D0059:L40。REG:L140 は −2.45、派生でも −2.45） | −0.96（REG:L140 は −0.97） | 下がる向きの「保留」帯。比較相手が基準＋EMA でないので暫定 | 0059（D0059:L35–43） |
| SiLU（vs Phase 1 seed 0、39.39） | 同上 | — | 未確認 | −1.06 | 同上 | D0059:L40 |

### 実行中・未完了（2026-10-16 00:48 JST、epoch は 0 始まり）

| run | ホスト | 最新 epoch | サブセット clean / PGD-10（最新 epoch） | 出典 |
|---|---|---|---|---|
| MobileNetV4-M 基準＋EMA | Hamster | 29（30/50） | 59.54 / 35.94（EMA PGD 36.94） | run:plan0103-phase2-mnv4m-baseline-ema-s0-v1 (H)。終了は 10/17 午前ごろの見込み（REG:L141） |
| EfficientNet-B0 基準＋EMA | Hamster | 15（16/50） | 40.62 / 23.34 | run:plan0103-phase2-effb0-baseline-ema-s0-v1 (H) |
| DeiT-Tiny 基準＋EMA | Ferret | 3（4/50） | 18.64 / 9.28 | run:plan0103-phase2-deit-tiny-baseline-ema-s0-v1 (F) |
| DeiT-Tiny cosine | Ferret | 3（4/50） | 18.70 / 9.28 | run:plan0103-phase2-deit-tiny-cosine-s0-v1 (F) |
| plan 0107 の3本 | Ferret | 下の6章 | | |

DeiT-Tiny の Phase 2 完了結果: なし。
キュー（未開始）: Hamster は MobileNetV4-M cosine、EfficientNet-B0 cosine。Ferret は DeiT-Tiny / MobileNetV4-M / EfficientNet-B0 のラベル平滑化 0.1、その後に β2 0.95 の4本（`queues/work_queue.txt`、両ホストで 2026-10-16 00:48 JST に確認）。

---

## 4. 「学習不足（underfitting）」の証拠

train probe = 拡張なしの学習画像 2,000 枚、評価モード、PGD-10（D0045:L45, D0047:L43）。

### MobileNetV4-S 基準

| 指標 | 値 | 出典 |
|---|---|---|
| probe clean / PGD-10 | 59.80 / 34.40 | D0045:L45, run:plan0103-phase2-mnv4s-baseline-ema-s0-v1 (H) |
| 25,620 枚 clean / PGD-10 | 53.85 / 30.71 | D0042:L33 |
| 5,000 枚サブセット（ep 49）clean / PGD-10 | 52.96 / 29.98 | run:plan0103-phase2-mnv4s-baseline-ema-s0-v1 (H) |
| probe と見ていない画像の差（PGD-10） | 4.4pt（REG:L133、P0107:L12。サブセットとの差に相当: 34.40 − 29.98 = 4.42 派生）。25,620 枚との差は +3.69（D0043:L45, D0044:L47） | |
| best epoch | 43（サブセットで選択、53.74 / 30.57）。Phase 1 は best = last（ep 49） | D0042:L35,L37, REG:L133 |
| best と last の差 | ImageNet の 29 本の中央値で 0.1pt | D0048:L45 |
| 学習損失（最終） | 3.697 | D0048:L44 |
| 学習画像への当てはまり | 学習に使った画像の PGD-10 も約 25〜33% で、完全な当てはまりまでの距離（約67〜75pt）が汎化の差（約1〜2.6pt）より遥かに大きい（Phase 1 前の測定、3モデル n=1） | P0103:L220–224 |

### AWP（MobileNetV4-S）

| 指標 | 基準 | AWP | 出典 |
|---|---|---|---|
| 最終の学習損失 | 3.697（REG:L133 では 3.70） | 5.640（REG:L133 では 5.64）。当てずっぽうは 6.91 | D0048:L44, REG:L133 |
| probe PGD-10 / サブセット PGD-10（ep 49） | 34.40 / 29.98 | 17.25 / 16.72 | run (H) の `train_probe_pgd_accuracy` / `val_subset_pgd_accuracy` |
| probe clean / PGD-10 | 59.80 / 34.40 | 27.55 / 17.25 | D0048:L44 |
| probe − 25,620 枚の差（clean / PGD） | +5.95 / +3.69 | −0.92 / −0.20 | D0048:L44 |
| 後半 25 epoch の学習損失 | — | 5.682 → 5.640 | D0048:L44 |
| LR 低下時の伸び（サブセット PGD-10） | ep 24→25: +5.18、ep 37→38: +0.50 | +0.98、−0.02 | D0048:L43 |

読み（記録）: AWP は頑健性の過学習への対策だが、基準にはその症状がない。差を縮めたのは当てはまりを下げたため（REG:L133）。正則化を強める手法（AWP、TRADES −4.65、IDBH）は下がり、最適化を助ける手法（SiLU +2.56、cosine +0.78）は上がる（REG:L133）。

### SiLU（MobileNetV4-S と MobileNetV4-M）

| 指標 | MNv4-S ReLU 基準 | MNv4-S SiLU | MNv4-M ReLU（Phase 1 seed 1） | MNv4-M SiLU |
|---|---|---|---|---|
| 最終の学習損失 | 3.697 | 3.608（D0044:L46） | 3.0232（run:plan0103-phase1-mobilenetv4-conv-medium-random-s1-v1 (H)） | 3.2404（run:plan0103-phase2-mnv4m-silu-s0-v1 (H)）。差 +0.217（D0059:L46） |
| 学習バッチの頑健精度 | 27.90% | 29.53%（D0044:L46） | 37.46%（run (H)） | 34.54%（run (H)）。差 −2.92pt（D0059:L46） |
| probe clean / PGD-10 | 59.80 / 34.40 | 62.10 / 38.25（run (H)） | 71.25 / 47.50（run (H)） | 66.70 / 44.00（run (H)）。差 −4.55 / −3.50（D0059:L48） |
| probe − 見ていない画像（PGD-10） | +3.69 | +4.98（D0044:L47） | +8.20（D0059:L48） | +5.67（D0059:L48） |
| 学習曲線 | — | 差は最初の epoch からあり続く（サブセット PGD-10: ep 9 +1.56、ep 24 +2.26、ep 49 +2.82）（D0044:L45） | — | 最初の 8 epoch は SiLU が損失が低い。epoch 8（LR 0.0225）で逆転、差は ep 24 +0.127、ep 49 +0.217（D0059:L45–46） |

MNv4-M SiLU の読み（記録）: ReLU 用の LR 0.025 のままの結果で、LR との相性が疑われる。悪ければ SiLU の LR 探索（最初は 0.0125）（REG:L141, P0107:L139–143）。

---

## 5. ConvStem の崩壊（debugging 0030）

- 何が起きたか: ConvStem 版 ConvNeXt-Atto（AdamW lr 1e-3、wd 0.05、β2 0.999）は 50/50 epoch「完了」したが、全精度 0.10%（当てずっぽう）、学習損失は epoch 0 から ln(1000) のまま。約 29 GPU 時間を使った。同じ source・レシピ・ホストの deep-narrow は 57.97 / 35.53（DBG30:L5–10）。崩壊の見張り（最高値の半分）は最高値が 0.1% のため発動しなかった（DBG30:L9–10）。
- 原因（CPU 調査）: 配線のバグではない。ランダム初期化の GELU / LayerNorm の conv stem は、stage-0 のエネルギーの約 40〜50% を入力に依存しない一定成分（DC）に置く（DBG30:L14–20）。

  | アーキテクチャ | stem 出力の DC 割合 | ダウンサンプルでの DC 割合 |
  |---|---|---|
  | ConvStem | 0.42 | 0.29–0.34 |
  | 標準 ConvNeXt-Atto | 0.003 | 0.003 |
  | deep-narrow | 0.004 | 0.004 |
  （DBG30:L22–26）

- 仕組み: step 0 からの PGD-AT で、勾配降下がこの DC を使ってネットワークを入力非依存にする（損失 ln(1000) の自明な頑健解）。epoch 0 の checkpoint で、入力依存の相対信号は stage 1〜3 の後で 1e-3、5e-5、2e-6。画像間の logit 標準偏差 3e-7（DBG30:L28–32）。
- 再現: ConvStem・PGD-AT は 0.163 → 0.025（25 step）で崩壊、clean CE では学習。DC を注入した Atto は崩壊（DC は十分条件）、初期 DC を引いた ConvStem は回復（DC は必要条件）（DBG30:L33–42）。weight decay は結果であって引き金ではない（stem の重みノルム 2.4 → 2e-4）（DBG30:L44–45）。
- 範囲: DeiT-Tiny ConvStem も同じ兆候（DC 0.50、25 step で崩壊）、Ferret のキューから外した。ols（DC 0.08）、MobileNetV4（BN が DC を除く）、Singh の教師モデルは影響なし（DBG30:L56–62）。
- 追加調査（なぜ Singh らのスクラッチ学習は成功したか）: Singh et al.（2303.01870、Table 1、random init., basic augment.）は ConvNeXt-T+CvSt 64.2 / 37.6 vs ConvNeXt-T 64.0 / 37.1、ViT-S+CvSt 62.8 / 34.4 vs ViT-S 61.5 / 31.8（DBG30:L81–83）。
  - 1要因ずつの CPU probe: ラベル平滑化 0.1、2 step 攻撃は効果なし。step ごとの LR ウォームアップは崩壊が深い。[0,1] 入力は悪化。Singh の幅の ConvNeXt-T+ConvStem（DC 0.42）も我々のレシピでは崩壊。**AdamW β2 = 0.95（Singh）vs 0.999（我々）だけが結果を変える**（DBG30:L87–94）。
  - 崩壊の始まりは全条件で起きる。β2 = 0.999 は 1,000 枚 probe で崩壊し続ける（logit std 0.163 → 0.0006、step 700）。β2 = 0.95 は step 150 で底を打ち 0.365 まで回復。DeiT-Ti ConvStem も同じ順序（DBG30:L96–99）。
  - 実 run の optimizer 状態: epoch 0 末で `stages.0.blocks.0.conv_dw` の sqrt(v) 中央値 4.9e-9（Adam eps 1e-8 未満）。epoch 48 で stem の更新比 約 3e-5（DBG30:L100–102）。解釈（未検証）: β2 0.999 では v が初期の大きな勾配を覚えて step が縮む（DBG30:L103–105）。未検証: バッチサイズ（Singh は約 1,392）（DBG30:L106）。probe の限界: 256 / 1,000 枚、1 seed、最大 700 step（DBG30:L108–109）。
- 対応: 「学習していない」学習を 2 epoch 以内に止める見張りを追加（650fbed）（REG:L139, DBG30:L65–69）。decision 0057 は B'（ConvNeXt-Atto と DeiT-Tiny で、β2 0.95 の ConvStem あり／なしを2本ずつ）（REG:L139）。

---

## 6. plan 0107（学習時だけ容量、デプロイは素のネットワーク）

- 承認: 2026-10-15（REG:L138, P0107:L3）。
- 仮説: 軽量ネットワークが敵対的学習で学習不足になるのは、主に、デプロイ時の関数クラスに含まれる良い解を最適化が見つけられないためで、関数クラスが小さすぎるためではない。正しければ、学習時だけ存在しデプロイ時のネットワークを変えない介入で、同じ計算量の大きいモデルとの差の大部分が縮むはず（P0107:L31–35）。
- 文献の根拠（P0107:L37–49）: Smooth AT（ImageNet ResNet-50、PGD-1）で SiLU ResNet-50 42.3 vs ReLU ResNet-152 39.4。ReLU 順方向＋滑らかな逆方向だけで 33.0 → 35.8（重み更新のみ）、36.9（攻撃と更新の両方）。学習 CE 2.71 → 2.59。MobileOne S0 +1.8。
- デプロイの制約: SE・sigmoid 系の活性化・attention は DSP/NPU で遅くなりうるので避ける。MobileNetV4 も同じ理由で SE と GELU を避ける（arXiv:2404.10518、6章）（P0107:L24–27, REG:L137）。

### 最初の3本（MobileNetV4-S、seed 0、50 epoch、非決定的、Ferret）

比較相手: 基準＋EMA 30.71、SiLU 33.27（P0107:L89）。判定は Phase 2 の3段階。SiLU の上がり幅（+2.56）の何割を取り戻すかも報告（P0107:L95–97）。attack_and_optimizer 版は学習攻撃が弱くなるので、学習損失・学習バッチ精度は比べず、probe と見ていない画像だけで比べる（P0107:L98–101）。

| run | 内容 | 状態（最新 epoch、0 始まり） | サブセット clean / PGD-10（最新） | 同じ epoch の基準＋EMA / SiLU（サブセット） |
|---|---|---|---|---|
| plan0107-mnv4s-hardswish-s0-v1 | hard-swish 順方向 | 実行中、epoch 15（16/50） | 38.48 / 21.30 | 37.88 / 19.76 / 39.36 / 21.78 |
| plan0107-mnv4s-smoothgrad-opt-s0-v1 | ReLU 順方向、softplus（β=10）の微分を重み更新だけに | 実行中、epoch 14（15/50） | 37.76 / 19.84 | 37.18 / 18.96 / 38.78 / 21.26 |
| plan0107-mnv4s-smoothgrad-both-s0-v1 | 同、学習攻撃と重み更新の両方に | 実行中、epoch 7（8/50） | 26.80 / 13.62 | 26.18 / 14.00 / 28.32 / 15.68 |

出典: plan 0107 の3本は run (F)（2026-10-16 00:48 JST）。比較列は run:plan0103-phase2-mnv4s-baseline-ema-s0-v1 (H) と run:plan0103-phase2-mnv4s-silu-s0-v1 (H) の同じ epoch の行。注意: 途中経過・サブセット 5,000 枚（1行ごとの差の SE は約 0.9pt、D0051:L49）・決定的（比較相手）と非決定的（新 run）の違いあり。判定は最終の 25,620 枚で行う。
実装コミット 8f51192、worktree source-8f511929ed7e（P0107:L135–137）。

### 次の手順（記録）

- hard-swish / smooth-backward の結果が先（DSP に優しい案を優先）。SiLU 並みの上がり幅なら、それをベースモデルにして蒸留へ。MobileNetV4-M もその方向。届かなければ MobileNetV4-M で SiLU の LR 探索（最初 0.0125）（P0107:L102–104, L139–143, REG:L141）。
- 蒸留検証と並行して、構造の再パラメータ化（MobileOne/RepVGG 型）を折りたたみ等価テスト付きで実装し、MobileNetV4-S（分岐あり）を同じ GPU 時間の MobileNetV4-M と比べる（P0107:L106–108）。
- 候補4: 蒸留段階で教師 bank の entropy / margin から「頑健には学べない」画像を見つけ、攻撃計算を配り直す（P0107:L68–80）。

### β2 0.95 の ConvStem 4本（キュー済み、未開始）

Ferret のキューの4〜7行目（ラベル平滑化3本の後）、worktree source-2455ece3e54b:
- plan0103-phase2-convnext-atto-baseline-ema-beta2-095-s0-v1
- plan0103-phase2-convnext-atto-convstem-beta2-095-s0-v1
- plan0103-phase2-deit-tiny-baseline-ema-beta2-095-s0-v1
- plan0103-phase2-deit-tiny-convstem-beta2-095-s0-v1

（Ferret `queues/work_queue.txt`、REG:L139、コミット 2455ece）。β2 は対応する基準にもかける必要がある（DBG30:L111–112）。

---

## 7. 蒸留の準備

| 項目 | 内容 | 出典 |
|---|---|---|
| 位置づけ | 新手法の提案ではなく、効くかどうかの検証。効かないことも成果 | REG:L124 |
| 教師（5つ） | Salman ResNet-50、Singh ConvNeXt-T ConvStem、Singh ViT-S ConvStem、Singh ConvNeXt-B ConvStem、手元の MobileNetV4-M（Phase 1） | REG:L126, REG:L136 |
| bank（確定） | FKD 方式、top-500（K=500）、FP16、50 epoch 分。2026-10-15 確定、manifest SHA-256 は REG:L136。Anteater で合計 593GB、Hamster・Ferret へコピー中 | REG:L135–136 |
| proxies v2 | 契約 `ard-distillation-proxies-v2`（RSLAD 攻撃、指標 A/B/C、同じレシピの教師）。実装はコミット 458b9ea。計算結果の数値は docs 内で未確認 | git 458b9ea のコミットメッセージ |
| 安い指標 | TAS、エントロピー、学習用切り出しでの教師の正解率（FKD 方式）を全5生徒×全教師で計算 | REG:L124 |
| 3×3 | 3生徒（MobileNetV4-S、ConvNeXt-Atto、DeiT-Tiny）× 3教師（同じ系統、強い別系統、弱い・中立）、50 epoch（30 epoch 時点の順位も記録）。EfficientNet-B0 と MobileNetV4-M は指標で選んだ教師1つ。最良の組だけ 100 epoch も | REG:L124 |
| 手法 | 主: RSLAD（教師はきれいな画像のみ）を FKD 方式で。比較: RSLAD-advT（目標を T(x′) に）。教師が毎 step 推論する普通の RSLAD は各生徒の最良教師で3組 | REG:L125 |
| GPU | 今のキューが空いたら 5 GPU（Hamster 2＋Ferret 3）で実行。順番: 蒸留検証 → 保留の深掘り → 確定したものの組み合わせ | REG:L135 |
| 教師の役割 | ConvNeXt-Atto と DeiT-Tiny の「強い教師」はどちらも ConvNeXt-B | REG:L126 |

---

## 8. 参考文献（IEEE 形式）

確認方法: arXiv API（export.arxiv.org、2026-10-16 取得）で題名・著者・初版年を照合。会場は arXiv の comment / journal_ref に書かれているものだけ「確認済み」。それ以外は【会場未確認】。

[1] D. Qin, C. Leichner, M. Delakis, M. Fornoni, S. Luo, F. Yang, W. Wang, C. Banbury, C. Ye, B. Akin, V. Aggarwal, T. Zhu, D. Moro, and A. Howard, "MobileNetV4: Universal models for the mobile ecosystem," arXiv:2404.10518, 2024.【会場未確認（arXiv に記載なし。ECCV 2024 とされるが未照合）】
[2] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. ICML*, 2019. arXiv:1905.11946.
[3] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," in *Proc. CVPR*, 2022. arXiv:2201.03545.
[4] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jégou, "Training data-efficient image transformers & distillation through attention," arXiv:2012.12877, 2020.【会場未確認（ICML 2021 とされるが未照合）】
[5] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, "Towards deep learning models resistant to adversarial attacks," in *Proc. ICLR*, 2018. arXiv:1706.06083.
[6] C. Xie, M. Tan, B. Gong, A. Yuille, and Q. V. Le, "Smooth adversarial training," arXiv:2006.14536, 2020.（arXiv の comment は "tech report"）
[7] N. D. Singh, F. Croce, and M. Hein, "Revisiting adversarial training for ImageNet: Architectures, training and generalization across threat models," in *Proc. NeurIPS*, 2023. arXiv:2303.01870.
[8] D. Wu, S.-T. Xia, and Y. Wang, "Adversarial weight perturbation helps robust generalization," in *Proc. NeurIPS*, 2020. arXiv:2004.05884.
[9] H. Zhang, Y. Yu, J. Jiao, E. P. Xing, L. El Ghaoui, and M. I. Jordan, "Theoretically principled trade-off between robustness and accuracy," in *Proc. ICML*, 2019. arXiv:1901.08573.
[10] A. Kurakin, I. Goodfellow, and S. Bengio, "Adversarial machine learning at scale," arXiv:1611.01236, 2016.【会場未確認（ICLR 2017 とされるが未照合）】
[11] C. Xie and A. Yuille, "Intriguing properties of adversarial training at scale," in *Proc. ICLR*, 2020. arXiv:1906.03787.（MBN）
[12] B. Zi, S. Zhao, X. Ma, and Y.-G. Jiang, "Revisiting adversarial robustness distillation: Robust soft labels make student better," arXiv:2108.07969, 2021.【会場未確認（ICCV 2021 とされるが未照合）】（RSLAD）
[13] L. Li and M. Spratling, "Data augmentation alone can improve adversarial training," in *Proc. ICLR*, 2023. arXiv:2301.09879.（IDBH）
[14] X. Ding, X. Zhang, N. Ma, J. Han, G. Ding, and J. Sun, "RepVGG: Making VGG-style ConvNets great again," in *Proc. CVPR*, 2021. arXiv:2101.03697.
[15] P. K. A. Vasu, J. Gabriel, J. Zhu, O. Tuzel, and A. Ranjan, "MobileOne: An improved one millisecond mobile backbone," in *Proc. CVPR*, 2023. arXiv:2206.04040.

使う可能性があるもの（同じ方法で照合済み）:
[16] A. Howard *et al.*, "Searching for MobileNetV3," in *Proc. ICCV*, 2019. arXiv:1905.02244.（hard-swish）
[17] C. Xie, M. Tan, B. Gong, J. Wang, A. Yuille, and Q. V. Le, "Adversarial examples improve image recognition," in *Proc. CVPR*, 2020. arXiv:1911.09665.（AdvProp、補助 BN）
[18] Z. Wang, X. Li, H. Zhu, and C. Xie, "Revisiting adversarial training at scale," in *Proc. CVPR*, 2024. arXiv:2401.04727.（AdvXL、2段階学習の元）
[19] T. Pang, X. Yang, Y. Dong, H. Su, and J. Zhu, "Bag of tricks for adversarial training," in *Proc. ICLR*, 2021. arXiv:2010.00467.
[20] F. Croce and M. Hein, "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks," in *Proc. ICML*, 2020. arXiv:2003.01690.（AutoAttack）
[21] H. Salman, A. Ilyas, L. Engstrom, A. Kapoor, and A. Madry, "Do adversarially robust ImageNet models transfer better?," in *Proc. NeurIPS*, 2020. arXiv:2007.08489.（教師 Salman ResNet-50）
[22] Z. Shen and E. Xing, "A fast knowledge distillation framework for visual recognition," arXiv:2112.01528, 2021.【会場未確認（ECCV 2022 とされるが未照合）】（FKD）
[23] M. Goldblum, L. Fowl, S. Feizi, and T. Goldstein, "Adversarially robust distillation," in *Proc. AAAI*, 2020. arXiv:1905.09747.（ARD）

（RobustBench は引用しない方針: REG:L80 H1）

---

## 未確認・ギャップ一覧

1. Phase 1 の seed 0 の clean のうち、MobileNetV4-M（63.79）、EfficientNet-B0（58.22）、ConvNeXt-Atto（59.96）、DeiT-Tiny（58.63）は docs に記録がなく、Ferret の run ファイルから直接読んだ。
2. EfficientNet-B0 と DeiT-Tiny の Phase 1 seed 1（58.31 / 35.76、57.10 / 33.63）は Ferret で完了しているが docs 未取り込み。seed 差（0.17、1.37）は派生値。DeiT の 1.37 は他のモデル（0.09〜0.36）より大きい。判定基準の見直しは未実施（REG:L129）。
3. ConvNeXt-Atto ols（59.29 / 36.45）は完了しているが record・decision packet がない。
4. DeiT-Tiny のベンチマーク速度（3設定の比較）は未確認。EfficientNet-B0 lr 0.025・ConvNeXt-Atto lr 1e-3・DeiT-Tiny lr 1e-3 の Phase 1 セルは Ferret 実行のため、Hamster での GPU 時間は未確認。
5. distillation proxies v2 の計算結果（数値）は docs 内に見つからない（実装コミット 458b9ea のみ確認）。
6. 未決の decision（chosen が null）: 0038、0039、0041、0042、0043、0044、0045、0046、0047、0052、0053、0054、0056、0058、0059。0042 の B（source 5dd329e が d836c03 と同じに学習するか）は未確認のまま。5dd329e の run（SiLU、IDBH、TRADES、SE など）と基準の比較はこの注意つき（例 D0044:L53–55）。参考: 標準×100 の run は基準と epoch 0〜7 が最後の桁まで一致した（D0054:L45）。
7. 文献の会場: [1][4][10][12][22] は arXiv 上に会場の記載がなく未照合。
8. MNv4-M SiLU の Δ clean は packet が −2.44、register が −2.45（丸め前の値の違い。表示値どうしの差は −2.45）。

---

## 追記（2026-10-18）：10/17〜18 に終わった run

出典はすべて run の `epoch-metrics.jsonl`（`val_full_*`、25,620 枚、最終エポック、普通の重み／EMA）。Ferret の run は Hamster の runs/ に rsync 済みで、記録への取り込み（postrun）は watcher が実行する。

| run | clean / PGD-10 | EMA | 比べる相手との差（PGD-10、派生） |
|---|---|---|---|
| MobileNetV4-M 基準＋EMA（Hamster） | 63.59 / 39.10 | 63.62 / 39.23 | Phase 1 seed 0 の 39.39 と −0.29。EMA は +0.13 |
| EfficientNet-B0 基準＋EMA（Hamster） | 58.11 / 35.87 | 58.30 / 36.07 | Phase 1 の 35.93 と −0.06。EMA は +0.20（記録 730befb） |
| DeiT-Tiny 基準＋EMA（Ferret） | 58.59 / 34.88 | 58.57 / 35.08 | Phase 1 の 35.00 と −0.12。EMA は +0.20 |
| DeiT-Tiny cosine 減衰（Ferret） | 60.36 / 36.26 | 60.32 / 36.34 | 基準＋EMA 比 +1.38（DeiT の seed の揺れ 1.37 と同程度） |
| MobileNetV4-S hard-swish（Ferret、非決定的） | 55.00 / 32.62 | 54.82 / 32.70 | 基準 30.71 比 **+1.91**（SiLU +2.56 の約 75%） |
| MobileNetV4-S 勾配だけ滑らか①（重みの更新だけ） | 54.37 / 29.49 | 54.49 / 29.75 | −1.22 |
| MobileNetV4-S 勾配だけ滑らか②（攻撃と更新） | 55.91 / 23.52 | 55.94 / 23.91 | −7.19（clean +2.06） |
| MobileNetV4-M SiLU（再掲） | 61.26 / 38.33 | — | Phase 2 の基準＋EMA（39.10）比 −0.77、clean −2.33 |

注意：plan 0107 の3本は非決定的モード、比べる基準と SiLU は決定的モード。seed 1本。学習損失・学習画像（probe）：hard-swish 3.678 / 60.10・36.45、①3.728 / 59.50・33.80、②3.535 / 61.40・27.40（② の学習中の攻撃は代理勾配で作るので、学習損失は他と比べない）。
