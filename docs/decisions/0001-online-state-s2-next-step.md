---
id: 0001
status: pending
created: 2026-09-05
campaign: ert-i100-online-state-s2-v1
question: Online-State S2×T1 preservation screen (plan 0087) が完了し PMP/DBDP とも両 seed で正だった。次の一手は何か。
options:
  A: PMP の確認実験。未使用 3 seed (confirm-a/b/c) の I100 e99 親から control vs OS-PMP を e101–114 で走らせ、e114 held-out CE-PGD20 で判定。約 6 arm × 28 min ≈ 3 GPU 時間 + endpoint。判定規則を事前登録（3/3 正かつ平均 ≥ 0.3 pp、noise floor 明記）。
  B: サンプル単位介入の系統を閉じ、branch C（Predictable but Not Necessarily Actionable）として I100 を主結果にする。I100 の official CIFAR-10 test + AutoAttack（best/last、5 seed 分の checkpoint 既存）を実施。GPU ≈ 5 seed × 2 ckpt × AA ≈ 10–15 時間。
  C: A と B を並列（Hamster で A、Ferret で B）。約 1 日で両方の証拠が揃う。
  D: 8-cell seed-0 コホートの multi-seed 化（protocol が求める 3 seed）。1 seed あたり 8 cell × 200 epoch ≈ 8 × 5 時間 = 40 GPU 時間/seed。論文の主表を固める。
recommendation: C
chosen: null
---

## 何が分かったか（plan 0087、記録は `docs/experiments/ert_rslad_i100_online_state_s2_preservation_v1.json`）

| 比較 | dev-1 Δ robust (e114, held-out CE-PGD20) | dev-2 Δ | 判定 |
|---|---|---|---|
| OS-PMP − Control | +0.14 pp | +0.20 pp | SUPPORTED（両 seed 正） |
| OS-DBDP − Control | +0.14 pp | +0.06 pp | SUPPORTED |
| OS-DBDP − OS-PMP | +0.00 pp | −0.14 pp | NOT_SUPPORTED |

- これはプログラム初の「arm vs control で SUPPORTED」判定。ただし plan 0079 の DPM − Control も +0.08/+0.12 pp で両 seed 正だった（当時の主比較が D-BDD vs DPM だったため未ラベル）。同方向の両 seed 正が 2 回。
- 効果量 0.06–0.20 pp は、shuffle/augmentation RNG だけで生じる局所変動 1–2 pp（`docs/ERT_RESEARCH_STATUS_SUMMARY.md`）の 1/10。2 seed の方向一致だけでは母集団主張にならない。
- コスト: PMP は epoch 時間 +1.2%、DBDP は +7.9%。DBDP に PMP を上回る根拠なし → 追うなら PMP のみ。
- ルーター品質は低い（online vs canonical S2×T1 の Jaccard 4.5–5.1%）が、それでも効果は出ている。
- 出自: arms/endpoints/canonical は attempt11 → recovery14/15/16 の 4 campaign で生成、集計は recovery17。全 20 個の checkpoint SHA-256 を照合済み。集計スクリプトの 2 バグ（attack identity 定数、parent SHA 欠落）は修正してコミット済み（66a223d）。記録は `docs/experiments/ert_rslad_i100_online_state_s2_preservation_v1.json`、報告は `docs/ERT_RSLAD_I100_ONLINE_STATE_S2_PRESERVATION.md`、plan 0087 の Completion report に注意点 (a)–(f) を記載。
- 契約上の stop: この screen 自体は e199 延長・seed 追加・official test を自動的には許可しない。次は新しい契約（= 本パケット）。

## 判断材料

- A は「PMP が本物か」を最小コストで答える。効果が 0.2 pp 級なら 3 seed でも検出力は不足しうるが、5 seed 全部で符号一致なら次段（e199 延長、official test）に進む根拠になる。
- B は修士論文の主結果を確定させる作業。I100 の +0.69 pp（3 unseen seed、内部 validation）はまだ official test/AA を通っていない。
- D は最も高価だが、8-cell 表を 1 seed のまま論文に載せるリスクを消す。A/B の後でも遅くない。

## 人間の記入欄

`chosen:` に A/B/C/D を書くか、チャットで指示してください。選択後、Claude が `/experiment-launch` 用の plan（新番号）を起こします。
