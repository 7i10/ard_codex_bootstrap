---
id: 0017
status: pending
created: 2026-09-22
campaign: plan0102-revisiting-at-recipe-full50-v1（decision 0016のオプションA実行後のフォローアップ、新しいGPUジョブはまだ無し）
question: decision 0016でオプションAを実施した結果、崩壊は本物（eval mode基準でロバスト精度は実質0%）であり、train_robust_accuracyの70.9%という数字は主にBatchNormのtrain/evalモード差によるものだったと判定された。原因（3-stepの弱い内側攻撃／AdamWのweight decay／重いデータ拡張）はまだ切り分けられていない。この続きとして、単一変数の切り分け（オプションB）、攻撃ステップ数を上げた新契約（オプションC）、それとも移植を打ち切る（オプションD）のどれを選ぶか。
options:
  B: 重いデータ拡張だけを外したアームを25 epoch走らせる。GPU約9.8時間
  C: 訓練時の攻撃ステップ数を3から上げた新しいcontractを立てて50 epoch走らせる。GPU約40〜50時間
  D: 移植を打ち切る。plain PGD-AT（Arm A）を現時点の最良レシピとして確定し、GPUはWorkstream Bか既存checkpointのAutoAttackに回す
recommendation: B
chosen: null
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

## 選択肢B: 重いデータ拡張だけを外したアームを25 epoch走らせる

**GPU時間**: 約9.8時間

**やること**: `imagenet_mobilenetv4_revisiting_at_recipe.yaml`から
`dataset.imagenet_heavy_augmentation`だけを`false`にした新しいconfigを
作り、25 epochだけ走らせる。他の材料（AdamW、学習率、weight decay、
label smoothing、weight-EMA、3-stepの攻撃）は据え置き。攻撃のidentityは
変えないので新しいprotocol idは不要（`tracking.group`とconfig名のみ追加）。

**何が分かるか**: 3つの容疑者（3-stepの弱い攻撃、AdamW/weight decay、
重いデータ拡張）のうち1つが単独で外れる。崩壊はepoch 20-21に出るので、
25 epochで判定がつく。今回はpart 1の観測性修正のおかげで、
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
必要になる。clean精度でArm Aに勝てるかどうかの判定には使えない
（n=1・25 epochでは数pt規模の優劣は読めない）。

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

---

## 推奨: B

理由は2つ。

第一に、decision 0016のオプションAの検査は「崩壊は本物」であることは
確認したが、3つの容疑者のどれが主因かは何も動かしていない。3つのうち
最も安く（約9.8 GPU時間）、かつ攻撃identityを一切変えずに切り分けられる
のがBです。Cは最も高く、しかもBで切り分けてからでも遅くない
（decision 0016自身が既にこの順序を推奨していた）。

第二に、Bは崩壊のタイミング（epoch 20-21）までに判定がつくので、
25 epochで止められる。もし重いデータ拡張が主因でなければ、残る2つの
容疑者（3-step攻撃、AdamW/weight decay）のうちどちらかを次に切り分ける
判断材料が増える。

**この推奨が変わる条件**: もし人間の判断としてWorkstream A自体を畳む
のであれば、素直にDを選ぶのが良い。逆にレシピ移植をどうしても救いたい
強い理由があるなら、Bを飛ばしてCに行くこと自体は筋が通るが、外れたときに
40〜50 GPU時間を失うことを承知の上で、という条件つきになる。
