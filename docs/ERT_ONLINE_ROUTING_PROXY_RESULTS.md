# ERT Dynamic Routing 前段診断結果

更新日: 2026-08-11  
Status: `CPU diagnostics complete; no training or routing intervention launched`

## 実行範囲

Chen ERTのL2/seed 1とL4/seed 2について、既存のCE-PGD20 strong replay、KL-PGD10
online `SampleStateStore`、既存CE/KL factorial artifactだけを読み込んだ。新しい攻撃、訓練、threshold tuning、official test、AutoAttackは実行していない。

固定した定義は、feature anchor `39/59/79`、terminal `189/194/199` のmajority
Future Failure、S1=robust-correctかつfragile q10以外、S2=robust-correctの
fragile q10、S3=robust-wrong、Teacher T1/T2/T3もpositive-margin下位q10である。
Top-Kは各anchorのCE-PGD20 current-correct cohort内で計算したanalysis-only値であり、
production thresholdには使わない。

## FF-count cohort

| run | anchor | current-correct cohort | FF count |
|---|---:|---:|---:|
| L2 | 39 | 22,772 | 342 |
| L2 | 59 | 22,466 | 391 |
| L2 | 79 | 22,536 | 411 |
| L4 | 39 | 22,405 | 374 |
| L4 | 59 | 22,308 | 430 |
| L4 | 79 | 22,798 | 352 |

Top-KのGT-count行では選択数をFF countと一致させたため、precisionとrecallは同じ値になる。
anchor 79のGT-count precision/recallは、L2でTeacher signed dominance `.691`、
Teacher clean-margin risk `.530`、online margin-EMA `.156`、online frequency `.156`、
strong Student margin `.136`。L4ではそれぞれ `.656`, `.526`, `.125`, `.136`, `.131`。
Teacher signalは強いが、これはCE-PGD20上の予測関連であり、KD介入の有効性や因果効果ではない。

## CE20 oracle と KL10 online proxy

strong Student margin riskとonline margin-EMA riskのSpearman相関は、L2で
`.882/.870/.851` (anchor 39/59/79)、L4で `.870/.869/.869`。robust correctness
agreementはL2 `.809/.803/.790`、L4 `.807/.798/.799`だった。

したがって、online historyはstrong oracleの順位情報をかなり保持する一方、stateの
hard classificationは完全一致ではない。特にrouting stateをそのまま置換する前に、
online側での校正と誤差許容を検証する必要がある。

状態混同行列（machine report `ce20_vs_online.state_confusion`）も保存した。これは
CE20 oracleのS1/S2/S3とKL10 onlineのS1/S2/S3を同一sample IDで集計したもので、
online stateをoracle stateへ置換したとは解釈しない。Top-Kの各行には予測対象との
Jaccardも併記し、固定qのmask比較とGT-count比較を分離した。

Teacherのonline forward primitiveは保存されていないため、online T1/T2/T3や
S×Tのonline一致率は算出していない。strong replayのTeacher stateをonline Student
stateと併記したセルは、Teacherがonline観測されたという意味ではない。

## S1×T3 と遷移

S1×T3は小標本だが両seedで高いFF率を示した。

| run | anchor | S1×T1 n / FF | S1×T2 n / FF | S1×T3 n / FF |
|---|---:|---:|---:|---:|
| L2 | 39 | 20,434 / 0.46% | 50 / 72.00% | 10 / 100.00% |
| L2 | 59 | 20,114 / 0.51% | 78 / 70.51% | 27 / 96.30% |
| L2 | 79 | 20,161 / 0.53% | 95 / 75.79% | 26 / 100.00% |
| L4 | 39 | 20,094 / 0.59% | 53 / 67.92% | 17 / 100.00% |
| L4 | 59 | 19,958 / 0.63% | 102 / 70.59% | 17 / 100.00% |
| L4 | 79 | 20,431 / 0.33% | 71 / 77.46% | 16 / 100.00% |

S1→S2/S3、S2→S3などの遷移行列はmachine reportに全件保存した。onlineの
39→59/59→79はoracleよりS2/S3遷移が多く、KL10 stateがCE20 stateより厳しい
状態を付ける傾向がある。これは「online proxyは使えるが、CE20 stateの完全な代替ではない」
という解釈を支持する。保存間隔は20 epochなのでflappingの1/2/3観測判定はできない。

## Attack objective comparison

既存factorialのterminal majorityを、CE-PGD20 oracleのFF maskと比較した。

| run | condition | FF rate | Jaccard vs CE20 oracle |
|---|---|---:|---:|
| L2 | CE-PGD10 | 26.55% | .967 |
| L2 | KL-PGD10 | 19.03% | .665 |
| L2 | KL-PGD20 | 20.04% | .681 |
| L4 | CE-PGD10 | 26.64% | .970 |
| L4 | KL-PGD10 | 18.36% | .644 |
| L4 | KL-PGD20 | 19.30% | .660 |

CE-PGD10がCE-PGD20 oracleに最も近く、KL-PGD10とKL-PGD20は相互には高い
Jaccard (`.944/.941`)だがCE系とは低い。今回のartifactではL2/L4ともCE-PGD20
factorial条件が欠落しているため、objectiveとstep数を完全な2×2で分離したとは
主張しない。ただし既存データ上は、step数よりCE/KL objective差の方がfailure
structureを大きく変えている可能性が高い。

## Delayed routing と次の判断

per-epoch online stateは保存されていないため、1-epoch delayed routing feasibilityは
blockedとした。39/59/79を1-epoch列として補間していない。

現段階の結論は、

1. student historyはonline routing proxyとして有望だが、CE20 oracleとのhard-state
   一致は約80%であり、そのままroute切替にはしない。
2. S1×T3は両seedで高FFだが小標本なので、protective interventionを自動起動しない。
3. CE attackはCE-defined failure stateとの整合が高い。RSLAD training innerをCEへ
   変更する短期pilotは、既存因果目的と別のmechanism pilotとして事前登録が必要。
4. 次に進むなら、per-epoch logging-only trajectoryを新規に収集し、online stateの
   calibration、transition stability、one-epoch delayを先に確認する。

## 再現性

実行コマンド:

```bash
PYTHONPATH=src /home/shunsukenaito/.conda/envs/adv/bin/python \
  -m ard.cli.ert_online_routing_proxy \
  --config configs/analysis/ert_online_routing_proxy_v1.yaml \
  --output-dir .cache/analysis/ert-online-routing-proxy-v1-final9
```

生成時のsource SHAは `6001f5d31a57a97f51e395cd2cfa5b49a598a38d`。
report SHAは `e2aaf3f7684fe86951aa8798c0d2d0e02a1977096bbf4584a2a347b7116a9e6a`、
lineage SHAは `d0dcfbc7b3737fd941ba35b27c5a60332abb43fbb7b4a283369d9129b04c0e34`。
完全なmachine reportは同じcache directoryにあり、入力ファイルのSHA、factorial条件の
attack/run/epoch/seed identity、欠落factorial artifactもlineageへ記録されている。
