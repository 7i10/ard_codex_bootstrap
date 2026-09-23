---
id: 0018
status: pending
created: 2026-09-23
campaign: plan0102-revisiting-at-recipe-no-heavy-aug-full50-v1（decision 0017オプションB実行後のフォローアップ。plan0102-r18-revisiting-at-recipe-full50-v1(オプションE)は継続監視中、まだ判定不能）
question: decision 0017のオプションBを実行した結果、重いデータ拡張を外しても崩壊はepoch 21で再発した（元の崩壊と同じepoch、同じ兆候：train_robust_accuracyがtrain_clean_accuracyを追い越し、val_pgd_accuracyがepoch9の半分を2epoch連続で下回った）。重いデータ拡張は原因ではないと判定された。残る容疑者は2つ：3-stepの弱い内側攻撃と、AdamWのweight_decay=5e-2。次にどちらを切り分けるか、あるいは両方並行するか。
options:
  F: weight_decayだけを下げた（または外した）アームを1本走らせる。GPU約9.8〜19.6時間、registryやprotocol変更不要
  C: 訓練時の攻撃ステップ数を3から上げた新しいcontractを立てて走らせる。GPU約40〜50時間、CLAUDE.mdルール6により新しいprotocol idが必要
  G: 両方を並行して走らせる（GPU0でF、GPU1の空きを待つか Bを早期終了してF、Eの結果を待ってからC）
  D: ここで打ち切り、plain PGD-AT（Arm A）を最良レシピとして確定する
recommendation: F
chosen: null
---

# 0018 revisiting_atレシピ崩壊、decision 0017オプションB実行後の続き

## 前提: 何が起きたか

decision 0017のオプションB（重いデータ拡張だけを外したアーム）を実行した結果、崩壊は防げませんでした。詳細は plan 0102 の Progress log
（2026-09-23のエントリ、コミット `dfce8fa`）にあります。

| epoch | train_clean | train_robust | val_clean | val_pgd | overtake |
|---:|---:|---:|---:|---:|---|
| 19 | 28.28% | 22.03% | 30.90% | 10.91% | False |
| 20 | 27.03% | 23.36% | 29.45% | 4.79% | False |
| 21 | 24.17% | **25.87%** | 29.33% | 4.43% | **True** |

崩壊のepoch（20-21）は元のアーム（重いデータ拡張あり）とほぼ同じで、兵候（train_robust_accuracyがtrain_clean_accuracyを追い越す、val_pgd_accuracyが epoch9基準値の半分を2epoch連続で下回る）も同じでした。decision 0017の事前登録ルールの両方の条件が破れたため、**重いデータ拡張は崩壊の必要条件ではなかった**と判定します。

残る容疑者は2つです。

1. 訓練時の内側攻撃が3ステップしかないこと（"fast-AT"に近く、壊滅的過学習の既知のリスク。Arm Aから変えていない値で、SGDの下では安全だった）
2. AdamW、weight_decay 5e-2（このMobileNetV4-Conv-Smallは3.77MパラメータのほとんどがBatchNorm主体の軽量 CNNで、標準的なImageNet ResNet系よりweight decayに敏感な可能性）

decision 0016のオプションA（フォワードのみの検査）とdecision 0017のオプションB（重い拡張の除去）はどちらも「崩壊は本物で、防げなかった」ことを確認しましたが、この2つの容疑者はまだ何も切り分けていません。

## GPU時間の見積り根拠

オプションBの実測（1409〜1497秒/epoch、平均約1480秒、Hamster 4090 1枚、world size 1、global batch 128）から。50 epochで約20.6時間。オプションCは内側攻撃のステップ数を3→10に上げるため、1バッチあたりforward/backwardが増え、実測の2〜2.5倍（約40〜50時間）を見込みます。

**GPUの空き状況**: Hamsterの両GPUは現在、decision 0017のB（GPU0、epoch21まで完了、残り約29 epoch、約12時間）とE（GPU1、epoch11まで完了、残り約39 epoch、約29時間）で埋まっています。新しいアームを今すぐ走らせるには、どちらかを早期終了するか、空くまで待つ必要があります。

---

## 選択肢F: weight_decayだけを下げた（または外した）アーム

**GPU時間**: 約9.8〜19.6時間（オプションBと同じ早期判定・早期停止ポリシー）

**やること**: `imagenet_mobilenetv4_revisiting_at_recipe.yaml`（重い拡張ありの元の崩壊アーム）から`optimizer.weight_decay`だけを変えた新configを作る。値の候補は、AdamWの一般的なデフォルト（1e-2やtorchvisionのrecipeが使う典型値）か、思い切って0.0（weight decayなし）。他の材料（3-stepの攻撃、heavy augmentation、label smoothing、weight-EMA）は据え置き。攻撃identityは変えないので新しいprotocol idは不要（`tracking.group`とconfig名のみ追加、同じprotocolを再利用）。`training.epochs: 50`のまま起動し、epoch 25あたりで早期判定する（decision 0017のBと全く同じポリシー）。

**何が分かるか**: 2つの容疑者のうち1つが単独で外れる。崩壊は epoch 20-21 に出るとわかっているので、その付近で判定がつく。

**事前登録する判定ルール**: epoch 25までに`val_pgd_accuracy`がepoch9相当の値の半分を下回らず、かつ`train_robust_accuracy_eval_mode`（decision 0016で追加した観測性）が`train_clean_accuracy`を明確に上回らなければ、「weight decayが崩壊の必要条件だった」と判定する。どちらか一方でも破れれば「weight decayは主因ではない」と判定し、残る唱一の容疑者（3-stepの攻撃）に絞る——その場合、オプションCを事実上確定させる根拠になる。

**リスク**: 外しても崩壊が残れば、容疑者は3-stepの攻撃だけになり、オプションCが唱一の残された切り分け手段になる（安く済ませられる可能性を使い切ってしまう）。

## 選択肢C: 訓練時の攻撃ステップ数を上げた新しいcontractを立てる

decision 0017・0016で既に書いた内容と同じ。GPU約40〜50時間、CLAUDE.mdルール6により新しいprotocol idが必要。**Fより先にCをやる理由は今のところ見当たらない**——Fの方5倍近く安く、しかも2つの容疑者のうち1つを機械的に除外できる。

## 選択肢G: FとCを並行、あるいはFをEの結果を待たず今すぐ

Hamsterの2GPUが両方ふさがっているため、Fを今すぐ走らせるには (a) オプションBを早期終了してGPU0を空ける、(b) オプションEの完走を待つ（約29時間）、(c) オプションBの残り約12時間の完走を待つ、のいずれかを選ぶ必要があります。**Bは既に判定材料（崩壊した、というepoch21の結果）を出し切っているので、残り29 epochを見る価値は「最終的な床の値」を知る程度に下がっています。** Fを急ぐ理由があるなら、Bを早期終了する（`best.pt`/`last.pt`/`epoch-metrics`は既存分だけで確定済みなので、早期終了しても記録は失われない）のが最も安い選択肢です。

## 選択肢D: ここで打ち切る

**GPU時間**: 約0時間。plain PGD-AT（Arm A）を最良レシピとして確定し、残りのGPUをWorkstream Bか既存checkpointへのAutoAttackに回す。本 plan の Workstream A はこれまでフル50 epochのアームを6本走らせ（weight-EMA 2種、TRADES+weight-EMA、sharp-temperature ADR、revisiting_atレシピ、そのheavy-aug ablation）、1本もplain PGD-ATの両軸を超えていません。

---

## 推奨: F

理由は2つ。第一に、FはオプションBの5分の1以下のコストで、残る2つの容疑者のうち1つを機械的に切り分けられる——Cを試す前にFをやらない理由がない（decision 0016・0017が既に確立した「安い方から先に」という順序をそのまま継続するだけ）。第二に、もしFで崩壊が防げれば、Cという最も高価な選択肢を試す必要すらなくなる。GPUを確保するために、判定材料を出し切ったオプションBを早期終了することも合わせて推奨します（オプションEはまだ判定不能なので継続）。
