---
id: 0017
status: pending
created: 2026-09-22
campaign: plan0102-revisiting-at-recipe-full50-v1（decision 0016のオプションA実行後のフォローアップ、新しいGPUジョブはまだ無し）
question: decision 0016でオプションAを実施した結果、崩壊は本物（eval mode基準でロバスト精度は実質0%）であり、train_robust_accuracyの70.9%という数字は主にBatchNormのtrain/evalモード差によるものだったと判定された。原因（3-stepの弱い内側攻撃／AdamWのweight decay／重いデータ拡張）はまだ切り分けられていない。この続きとして、単一変数の切り分け（オプションB）、攻撃ステップ数を上げた新契約（オプションC）、それとも移植を打ち切る（オプションD）のどれを選ぶか。
options:
  B: 重いデータ拡張だけを外したアーム。50 epoch設定のまま早期判定、必要なら早期停止。GPU約9.8〜19.6時間
  C: 訓練時の攻撃ステップ数を3から上げた新しいcontractを立てて50 epoch走らせる。GPU約40〜50時間
  D: 移植を打ち切る。plain PGD-AT（Arm A）を現時点の最良レシピとして確定し、GPUはWorkstream Bか既存checkpointのAutoAttackに回す
  E: 同じレシピのままアーキテクチャだけ入れ替えたcontrol armをもう1枚のGPUで並走させる（ResNet-18かResNet-50、要選択）。GPU約34.8〜80時間
recommendation: B（今回GPU0で即実行）と、E（GPU1で並走、アーキテクチャ選択は本文参照）
chosen: B
---

# 0017 revisiting_atレシピ崩壊、decision 0016オプションA実行後の続き

## 前提: decision 0016で何をして、何が分かったか

decision 0016（`docs/decisions/0016-*`、`chosen: A`）で選んだオプションAを
2026-09-22に実行しました。詳細は plan 0102 の Progress log
（2026-09-22のエントリ、コミット `0a96e1c`）にあります。

**Part 1（観測性の修正、コミット`0a96e1c`）**: `train_robust_accuracy`と
mode一致した`train_robust_accuracy_eval_mode`（clean精度と同じ、post-step・
eval modeのforward）と、記録専用の`train_robust_overtakes_clean`フラグを
追加しました。既存の`train_robust_accuracy`の定義・値・利用箇所（
`gap_adaptive_step`など）は一切変えていません。

**Part 2（フォワードのみの検査、GPU約1回分、訓練なし）**: 崩壊した
`plan0102-revisiting-at-recipe-full50-v1`の`last.pt`（epoch 49）に対して、
同じデータセット設定から新しく1バッチ（128枚）を取り、eval mode・train mode
それぞれでclean精度と攻撃後精度を測りました。

| | clean | 3-step攻撃後 | 10-step攻撃後 |
|---|---:|---:|---:|
| eval mode | 28.12% | **0.00%** | 0.00% |
| train mode | 33.59% | 77.34% | 68.75% |

decision 0016で事前登録していた判定ルールが発動しました。eval mode(c)と
train mode(d)の差が77.34ptで、20pt以上という基準を大きく超えています。
つまり「見出しの70.9%は主としてmodeの非対称による見かけ」という判定です。
公式の評価プロトコルと同じeval modeで見ると、このモデルは3-step攻撃でも
10-step攻撃でも**ロバスト精度0%**でした。これは学習中に記録されていた
`val_pgd_accuracy=0.05%`とほぼ一致します。

## この検査で何が変わり、何が変わらないか

**変わらないこと**: 崩壊は本物です。eval mode（本番の評価と同じ条件）で
ロバスト精度が実質ゼロなのは測定artifactではなく、実際にモデルが頑健性を
何一つ獲得できなかったことを意味します。postrun記録の「clean 28.6% /
PGD-10 0.05%、Arm Aに対して大敗」という結論はそのままです。

**変わること**: 崩壊の「原因」の解釈です。postrunの記録
（コミット`015bc8d`）は、保存済みpanelデータ（`panel-epoch-49.jsonl`）の
「clean画像で間違える15サンプルのうち10サンプルが攻撃後に正解に変わる」
という現象を「壊滅的過学習とラベルリーク」の決め手として挙げていました。
しかしそのpanelの予測は`trainer.py:1541-1542`のコードにより
train mode・pre-stepのforwardから来ています。今回eval modeで見ると
ロバスト精度は一律0%なので、train modeで見えていた「一部のサンプルが
攻撃で正解に変わる」という選択的なパターンが、eval modeでも同じ形で
再現するかは未検証のままです。したがって「ラベルリークという具体的な
メカニズム」はやや弱い根拠になり、確実に言えるのは「モデルは全く
頑健でない」という、より単純だが同じくらい深刻な結論だけです。

**decision 0016のルール上、オプションCの事前確率は上がりませんでした。**
「(c)≈(d)かつ10-step攻撃がほぼ0」だった場合にのみCの事前確率を上げる
という規則でしたが、今回は(c)と(d)が大きく異なったため、この分岐は
発動していません。次に何を選ぶかについて、この検査は直接の証拠を
追加していません。

## ノイズ下限について

今回問題になっている差（clean 26pt、PGD 30pt、eval mode攻撃後精度0%）は
CIFARの参照ノイズ下限（0.16〜1.88pt）よりも桁違いに大きく、「崩壊したか
どうか」の判定にノイズ下限は制約になりません。B・Cのどちらも「崩壊するか
否か」の判定に使うのであれば同様です。

## GPU時間の見積り根拠

decision 0016と同じ実測値（1409秒/epoch、Hamster 4090 1枚、world size 1、
global batch 128）から。25 epochで約9.8時間、50 epochで約19.6時間。
オプションCは内側攻撃のステップ数を3→10に上げるため、1バッチあたりの
forward/backwardが増え、実測の2〜2.5倍（約40〜50時間）を見込みます。
Hamsterは現在2枚とも空いています。

---

## 選択肢B: 重いデータ拡張だけを外したアーム

**GPU時間**: 約9.8〜19.6時間（下記の通り、早期判定で打ち切る可能性あり）

**やること**: `imagenet_mobilenetv4_revisiting_at_recipe.yaml`から
`dataset.imagenet_heavy_augmentation`だけを`false`にした新しいconfigを
作る。**`training.epochs`は50のまま変えない**——`warmup_cosine`の減衰は
`total_epochs`に対する形なので、epoch数だけ25に短縮すると同じ50-epoch
アームの前半25 epochとはLRの下がり方が変わってしまい、「重いデータ拡張の
有無」以外の変数も動かしてしまう。本planが既に採用している方針
（2026-09-15決定：50 epoch設定のまま起動し、中間のepoch-metricsで
early go/no-goを判定し、望み薄なら早期に打ち切る）をそのまま適用する。
他の材料（AdamW、学習率、weight decay、label smoothing、weight-EMA、
3-stepの攻撃）は据え置き。攻撃のidentityは変えないので新しいprotocol id
は不要（`tracking.group`とconfig名のみ追加、同じprotocol
`controlled_imagenet_stage01_mobilenetv4_revisiting_at_recipe_v1`を再利用）。

**何が分かるか**: 3つの容疑者（3-stepの弱い攻撃、AdamW/weight decay、
重いデータ拡張）のうち1つが単独で外れる。崩壊はepoch 20-21に出るので、
epoch 25あたりで判定がつく。今回はpart 1の観測性修正のおかげで、
`train_robust_accuracy_eval_mode`と`train_robust_overtakes_clean`を
epoch単位で見れば、崩壊の兆候をtrain modeの見かけの数字に惑わされずに
早期発見できる。

**事前登録する判定ルール**: epoch 25までに
`val_pgd_accuracy`がepoch 9の値（13.59%）の半分を下回らず、かつ
`train_robust_accuracy_eval_mode`が`train_clean_accuracy`を一度も
明確に上回らなければ、「重いデータ拡張が崩壊の必要条件だった」と判定する。
どちらか一方でも破れば「重いデータ拡張は主因ではない」と判定し、
残る2つの容疑者に絞る。

**リスク**: 外しても崩壊が残れば容疑者が2つ残り、もう1本（約10 GPU時間）
必要になる。early go/no-goで「崩壊しなさそう」と判定して最後まで走らせても、
clean精度でArm Aに勝てるかどうかの判定にそのままは使えない
（n=1では数pt規模の優劣は読めない）。

## 選択肢C: 訓練時の攻撃ステップ数を上げた新しいcontractを立てる

**GPU時間**: 約40〜50時間

**やること**: 訓練時の攻撃ステップ数はCLAUDE.mdルール6の保護対象なので、
既存configの編集ではなく新しいscientific contractとして、新しい
protocol idと新しいconfig、独立したplanエントリを立てる。壊滅的過学習に
対する文献上の標準的な対処。

**何が分かるか**: 「3ステップという弱い内側攻撃が崩壊の原因か」が決まる。
当たればレシピ移植そのものを救える可能性がある。ただし今回の検査は
この仮説を直接支持も否定もしていない（eval modeで見ても崩壊は既に
3-step時点で完全なので、"攻撃が弱すぎて訓練中にモデルが気づかず過適合した"
という機序自体は依然としてあり得る、が確認はできていない）。

**事前登録する判定ルール**: 選択肢Bと同じ2条件をepoch 25で適用する。
さらに50 epoch完走時に`val_pgd_accuracy`がArm Aの30.6%を下回らなければ
「レシピはステップ数を上げれば機能する」と判定する。下回る場合、差が
2pt未満なら「判定不能」とし、勝ったとは書かない。

**リスク**: 一番高価。Bをやる前にCに行くと、外れたときに40〜50 GPU時間を
捨てることになる。Arm Aとの比較は「訓練レシピの比較」ではなく「攻撃強度も
変えた比較」になる点を新しいplanに明記する必要がある。

## 選択肢D: 移植を打ち切る

**GPU時間**: 約0時間

**やること**: revisiting_atレシピの移植を打ち切り、plain PGD-AT（Arm A）を
現時点で最良の訓練レシピとして確定する。HamsterのGPUはWorkstream B
（軽量ImageNet向けARD後継）か、既存checkpointへのAutoAttack
（decision 0014がまだ`pending`）に回す。

**何が分かるか**: 新しいことは何も分からない。本planのWorkstream Aは
これまでフル50 epochのアームを5本走らせ（weight-EMA 2種、
TRADES+weight-EMA、sharp-temperature ADR、revisiting_atレシピ）、
1本もplain PGD-ATの両軸を超えていない。今回は超えなかっただけでなく
崩壊した。打ち切りは現実的な選択肢。

**事前登録する判定ルール**: 判定するものがないため、ルールなし。資源配分の
判断。

**リスク**: 原因を記録しないまま閉じると、将来別のアームで同じ組み合わせ
（3-step攻撃＋重い拡張＋強いweight decay）を再現し、同じ崩壊を繰り返す
おそれがある。

## 選択肢E: 同じレシピのままアーキテクチャだけ入れ替えたcontrol arm

人間からの提案（chat, 2026-09-22）: revisiting_at論文自体の再現を、もう1枚の
Hamster GPU（GPU1、現在idle）で並走させ、「原因の一つがモデル（軽量アーキ
テクチャ）自体である」という可能性を切り分けたい、というものです。

**論文をそのまま再現するのではなく、レシピを固定してアーキテクチャだけ
変える方を推奨します。** 理由は、Singh/Croce/Hein 2023自身の学習コード
（`revisiting-at/main.py`、このplanで既に読んでいます）は内側の攻撃に
**APGD**（AutoAttackのAuto-PGD、この project の `LinfPGD` とは別アルゴリズム）
を使っています。論文を文字通り再現しようとすると、新しい攻撃アルゴリズムを
`src/ard/attacks/`に追加する必要があり、それ自体が新しい科学的contractで
scientific-reviewerのレビューが要り、しかも「アーキテクチャ」と「攻撃
アルゴリズム」という2つの変数が同時に変わってしまい、目的（モデルが原因か
の切り分け）にとってはむしろ不利です。代わりに、**今回崩壊した
`imagenet_mobilenetv4_revisiting_at_recipe.yaml`を一切変えず、
`student.architecture`だけを差し替える**方が、変数が1つだけ動く、
安く済む、レビュー不要（新しい攻撃も新しいmethodも追加しない）という
3拍子が揃います。

**アーキテクチャの選択が実測コストに直結する、人間が決めるべき分岐点です**：

| | パラメータ数 | pretrained対応 | このplanでの実測/epoch | 50 epoch見積り |
|---|---:|---|---:|---:|
| `resnet18_imagenet` | 11.2M | 既に対応済み（plan 0100で追加、registry変更ゼロ） | 実測2503秒（plan 0100 `r18_baseline-s0`、Hamster 4090、batch 128、この project 自身のPGD-AT） | 約34.8時間 |
| `resnet50_imagenet` | 25.6M | 未対応（`resnet18_imagenet`と全く同じパターンで`pretrained=True`分岐を追加する必要あり、小さく前例のある変更） | 実測なし。FLOPs比（ResNet-50/ResNet-18 ≈ 2.3倍）からの推定 | 約80時間（推定、実測ではない） |

ResNet-50はSingh/Croce/Hein論文が主に報告している規模により近く、「論文が
実際に検証したような、軽量でない標準アーキテクチャ」としての説得力は
ResNet-18より高いです。一方ResNet-18は、①実測データが既にある、②registry
変更が要らない、③GPU時間が半分以下、という理由でリスクが低いです。
どちらも「壊滅的過学習が起きやすいのは軽量モデル固有か」という仮説に対する
controlとしては機能しますが、ResNet-18は「中容量」（MobileNetV4の約3倍）
にとどまり、ResNet-50ほど明確な対比にはなりません。

**GPU時間**: ResNet-18なら約34.8時間、ResNet-50なら約80時間（推定）。
どちらもオプションB（GPU0側）と並行して、もう1枚のGPUで動かせます。

**やること**（アーキテクチャ確定後）: `student.architecture`と
`tracking.group`だけを変えた新configを作る。ResNet-50を選ぶ場合のみ、
`resnet18_imagenet`と同じパターンで`registry.py`に`pretrained=True`分岐を
追加する小さな変更が先に必要（新しい攻撃・objective・チェックポイント
機構には触れないので、レビューなしで進められる範囲だが、`scripts/verify.py
--changed`は通す）。攻撃identityは一切変えないので新しいprotocol idは
不要（現在の`controlled_imagenet_stage01_mobilenetv4_revisiting_at_recipe_v1`
とは学生アーキテクチャが違う一群として、`tracking.group`のみ変えて区別）。
このアームも50 epoch設定のまま起動し、Bと同じ早期判定ポリシーを適用する。

**何が分かるか**: 同じ崩壊のタイミング（epoch 20-21付近）・同じ形（train
robustがclean を追い越す、val_pgdがゼロへ滑落）が標準容量のアーキテクチャ
でも再現すれば、「軽量アーキテクチャ固有」という作業仮説は否定され、
このレシピ自体（3-step攻撃、AdamW+wd、重い拡張のどれか）が原因という
方向に絞られる。逆に標準アーキテクチャでは崩壊が起きなければ、「軽量モデル
は壊滅的過学習を起こしやすい」という作業仮説を支持する材料になり、その後の
独自軽量向け手法の設計（データ拡張→アーキテクチャ→ロス関数の優先順位、
2026-09-19の人間の方針）に直接効いてくる。

**事前登録する判定ルール**: epoch 25までに、Bと同じ2条件
（`val_pgd_accuracy`がepoch 9相当の値の半分を下回らず、かつ
`train_robust_accuracy_eval_mode`が`train_clean_accuracy`を明確に
上回らない）が両方満たされれば「このアーキテクチャでは崩壊しない」と判定
する。どちらか一方でも破れれば「このアーキテクチャでも崩壊する」と判定し、
軽量モデル固有という仮説を後退させる。

**リスク**: 一番高価な選択肢（ResNet-50なら最も高い）。B・Cとは独立した
軸（アーキテクチャ）を動かすので、Bの結果と組み合わせて初めて「重い拡張は
軽量モデルだけで壊滅的過学習を起こすが標準モデルでは起こさない」といった
複合的な結論が言えるようになる——単独では「原因のうち何が」までは絞れず、
「軽量アーキテクチャが必要条件かどうか」だけを教える。

---

## 推奨: BをGPU0で即実行、EをGPU1で並走（アーキテクチャは要相談）

Bについての理由は2つ。

第一に、decision 0016のオプションAの検査は「崩壊は本物」であることは
確認したが、3つの容疑者のどれが主因かは何も動かしていない。3つのうち
最も安く（早期停止すれば約9.8 GPU時間）、かつ攻撃identityを一切変えずに
切り分けられるのがBです。Cは最も高く、しかもBで切り分けてからでも遅くない
（decision 0016自身が既にこの順序を推奨していた）。

第二に、Bは崩壊のタイミング（epoch 20-21）までに判定がつくので、
早期に止められる。もし重いデータ拡張が主因でなければ、残る2つの
容疑者（3-step攻撃、AdamW/weight decay）のうちどちらかを次に切り分ける
判断材料が増える。

Eについては、B・Cとは直交する軸（アーキテクチャ）を同時に動かせるので、
GPU1が空いている限り並走させない理由がありません。ただしアーキテクチャ
（ResNet-18 vs ResNet-50）はコストとpaper-fidelityのトレードオフで、
どちらが正しいかを機械的には決められないため、人間の判断を仰ぎます。

**この推奨が変わる条件**: もし人間の判断としてWorkstream A自体を畳む
のであれば、素直にDを選ぶのが良い。逆にレシピ移植をどうしても救いたい
強い理由があるなら、Bを飛ばしてCに行くこと自体は筋が通るが、外れたときに
40〜50 GPU時間を失うことを承知の上で、という条件つきになる。
