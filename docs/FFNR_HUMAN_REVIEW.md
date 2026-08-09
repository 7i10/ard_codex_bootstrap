# FFNR盲検画像レビュー

## 起動

Formal panelからHTMLを生成します。既存HTMLを再生成する場合だけ`--force`
を付けてください。HTML、画像、manifestは同じartifactディレクトリに置くと、
画像の相対パスがそのまま解決されます。

```bash
PYTHONPATH=src /home/shunsukenaito/.conda/envs/adv/bin/python \
  scripts/render_ffnr_review.py \
  --manifest .cache/analysis/ffnr-strong-diagnostics-6a90011-v1/ffnr-strong-blinded-candidates.json \
  --output .cache/analysis/ffnr-strong-diagnostics-6a90011-v1/ffnr-human-review.html
```

生成済みHTMLは次です。

```text
.cache/analysis/ffnr-strong-diagnostics-6a90011-v1/ffnr-human-review.html
```

ブラウザで直接開くか、GUIのないサーバーではartifact directoryをHTTP経由で
閲覧します。判定はブラウザの`localStorage`へ自動保存され、`判定をJSON保存`
で別ファイルへexportできます。別ブラウザ・別PCで続ける場合は、そのJSONを
`JSON読込`からimportしてください。manifest SHAが異なるJSONは拒否されます。

## 判定カテゴリ

目的は「画像と、表示されたCIFAR-10の提示ラベルが整合しているか」を記録することです。
訓練中の失敗理由、teacherの正誤、target/controlの役割は判定しません。

| Category | 使う条件 |
| --- | --- |
| `clear_match` | 提示ラベルの対象が画像に明確に写っている。小さくても対象が確認できる場合を含む。 |
| `ambiguous` | ぼけ、遮蔽、極端な構図などで判断しにくいが、提示ラベルとの明確な矛盾まではない。 |
| `possible_label_error` | 画像内容が明確に別クラスに見え、提示ラベルとの不一致を疑う。単にrobustに難しいだけでは選ばない。 |
| `ungradable` | 画像破損、ほぼ一様、対象が完全に見えないなど、画像から判定できない。 |

`possible_label_error`は強い主張なので、迷う場合は`ambiguous`にしてください。
候補クラスは任意入力であり、確信がなければ空欄のままにします。確信度は
画像と提示ラベルの判定に対する確信で、`possible_label_error`でなくても低くできます。
メモには見えている根拠だけを書き、モデルの予測や実験結果を推測して書きません。

## 推奨レビュー手順

1. 全200枚を一度通し、各画像を4カテゴリのいずれかへ分類する。
2. `ambiguous`と`possible_label_error`だけをfilterして再確認する。
3. 可能なら、二人目がsample IDの順番を変えて独立に判定する。
4. 一人だけの判定では、`possible_label_error`を確定ラベルではなく「人手確認候補」として扱う。
5. 結果を見て自動的に介入対象や学習データ除外を決めず、次のscreenの事前計画へ反映する。

HTMLの表示順はrole-independent hash順で固定されています。manifestには画像、
提示ラベル、panel/run識別子だけが入り、outcome、score、teacher state、target/control
roleは含まれません。現在のmanifest SHA-256は
`8d1f21bec9dae9c4d750693374f6b0fd0f752db512491e87e10ac227608a8a02`です。
