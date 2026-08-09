# FFNR人手画像レビュー結果

## 入力整合性

- 対象manifest SHA-256: `8d1f21bec9dae9c4d750693374f6b0fd0f752db512491e87e10ac227608a8a02`
- 人手判定JSON SHA-256: `c5b5feec28fd9c3b591288b44e59ef8dca6cb903864961e70d9939f12dbe9bf5`
- 判定件数: `200/200`
- `clear_match`: `175` (`87.5%`)
- `ambiguous`: `25` (`12.5%`)
- `possible_label_error`: `0`
- `ungradable`: `0`
- confidence: high `199`、medium `1`、low `0`

manifest SHAが一致しているため、別panelの判定結果を誤って結合していません。
判定JSONのローカル保存先は、ignored cache内の次です。

```text
.cache/analysis/ffnr-strong-diagnostics-6a90011-v1/human-review-result.json
```

## run別の分布

| run label | clear_match | ambiguous | total |
| --- | ---: | ---: | ---: |
| L2 | 88 | 12 | 100 |
| L4 | 87 | 13 | 100 |

`possible_label_error`が0件だったため、今回のpanelには「提示ラベルが明らかに
間違っている」と人手で判断された例はありません。したがって、persistent-wrong
の主因をラベルノイズと結論づける根拠は得られていません。

## 解釈

今回の`ambiguous`は、ユーザー定義どおり「人の目でも判別しにくい」例です。
これはラベル誤りではなく、低解像度、遮蔽、非典型的な視点、背景情報の少なさ、
dog/catやautomobile/truckのような近接クラスの混同可能性を表します。

一方、`clear_match`の中にも「小さい」「珍しい姿勢・色」「典型的でない構図」
というコメントがありました。したがって今後は、少なくとも次を分けて扱います。

1. clear and easy: 明確で追加コメントなし
2. clear but hard: ラベルは明確だが、モデルには難しそう
3. human ambiguous: 人にも判別困難
4. suspected label error: 今回は該当なし

現在のJSONは1と2を`clear_match`にまとめ、3を`ambiguous`にしています。これは
人手レビューの再現可能な一次記録として保持し、訓練targetのhard labelを変更する
根拠にはしません。

## 研究上の扱い

- この結果は、persistent-wrong群に明白なラベル誤りが大量に含まれるという仮説を支持しない。
- `ambiguous`は、boundary/attackabilityの難しさと整合するが、原因を証明しない。
- `clear_match`でもhardな例があるため、画像の明瞭さだけでTeacher KDの有用性は判断できない。
- 次のscreenでは、hiddenなpersistent/recovered/control群へこの人手判定を後からjoinし、群ごとの`ambiguous`率を比較する。
- その比較を見る前に、介入routeやデータ除外を決めない。`possible_label_error`が0件でも、画像難度への介入効果は別途検証が必要である。

判定手順とカテゴリ定義は[FFNR_HUMAN_REVIEW.md](FFNR_HUMAN_REVIEW.md)に固定しています。
