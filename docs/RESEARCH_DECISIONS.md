# Research decisions

## D1. Single robust teacherを主設定にする

主研究質問は、教師のoverconfidenceと学生固有の時変robust learnabilityの相互作用です。dual-teacherを主経路にすると、clean/robust teacher間の勾配競合という別の交絡要因が入ります。そのためCIARD/B-MTARDは関連研究・任意比較とし、主実装には含めません。

## D2. RSLADを開発基盤、SAADを必須比較対象にする

RSLADはteacher-forward中心の軽量なsingle-teacher基盤として、signal/policyの効果を分離しやすい利点があります。SAADはentropy-onlyの最重要比較対象です。full SAADは強力ですが複数要素を含むため、初期開発基盤にはしません。

比較の最小構成:

1. RSLAD
2. RSLAD + SAAD entropy weighting
3. RSLAD + student robust learnability
4. RSLAD + joint risk
5. full SAAD（upstream reproductionを含む）

## D3. 学習不可能サンプルを削除しない

robust learnabilityは学生、学習時点、脅威モデルに依存します。サンプル自体を永久除外せず、高リスク時にteacher KDを弱め、hard-label adversarial objectiveへfallbackします。

## D4. 最初の学生信号はrobust-margin EMA

実装・計算コスト・解釈性のバランスからrobust-margin EMAを最初に採用します。robust correctness frequency、forgetting、checkpoint disagreementは同じinterface上の後続候補とします。

## D5. ImageNetは初期スコープ外

CIFAR-10/100で機構と再現性を確立し、Tiny-ImageNetをscale validationとします。ImageNet-100/1Kは、計算効率と再現性が確認された後の条件付き拡張です。

## D6. 全production experimentをW&Bへ保存する

定量指標だけでなく、固定sample panel、entropy、student margin、joint risk、weight、clean/adv予測をW&B Tablesで比較可能にします。ネットワーク不安定時はoffline-syncを許可しますが、同期完了までproduction runを完了扱いにしません。

## D7. テストを重複実行しない

Git diffと関連入力hashで必要testを選び、成功結果をcacheします。full suiteはmilestone境界、GPU smokeは影響pathがある場合、production trainingは明示的な研究実験としてのみ実行します。

### M0 target softening decision

`teacher_target_uniform_mix@1` は teacher の `softmax(z_t/T)` を adversarial student-KD branch のみ uniform mix する（`risk_transform: identity`, `rho_max: 0.5`）。clean KD target は不変で、student/joint main semantics に hard-label fallback はない。旧挙動は `rslad_hard_fallback@1` ablation として明示する。
