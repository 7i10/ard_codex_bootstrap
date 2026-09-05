---
name: biweekly-review
description: Close finished confirmations into the records, refresh the status summary, and draft the biweekly progress deck. Use every two weeks before a lab progress presentation, or whenever a confirmation completes.
---

# 隔週レビュー

締め切りまで「改善を続け、迫ったら打ち切る」進め方を成立させるための唯一の仕組み。
**目的は、いつ打ち切っても書ける状態を保つこと**であり、進捗を飾ることではない。

順に実行する。前の段が終わるまで次に進まない。

## 1. 完了した確定を記録に閉じる

`scripts/ardx/status.py --brief` と `--inventory` で、完了しているのに
記録に落ちていない campaign を洗い出す。完了判定は完了契約に従う。

- orchestrator campaign: `state.json` の `jobs[*].status` が全て TERMINAL
- 手動ラン: `run-bundle/completion.json` と `manifest.status ∈ {completed, sync_pending}`
  の両方、かつエラーマーカーが "no application error recorded"

該当があれば、対応する aggregator を回して結果記録（JSON + Markdown + `.sha256`）を
生成し、プランに完了報告を書いて閉じる。**ここを飛ばすと、打ち切り時に
その回の作業が丸ごと失われる。**

## 2. status summary を更新する

`docs/ERT_RESEARCH_STATUS_SUMMARY.md` の該当行を、1 で作った記録に合わせて書き換える。

- 記録に無い数値を書かない
- 効果量には必ず床（`docs/MEASUREMENT_STANDARD.md` §5）との対比を添える。
  床の内側なら「床の内側である」と明記する
- 採択・不採択は `docs/MEASUREMENT_STANDARD.md` §5.4 の規則にのみ従う

## 3. スライド内容を書く

`reports/biweekly/<YYYY-MM-DD>.md` に、`scripts/ardx/build_slides.py` の
文法で内容を書く（`# 題`, `## 見出し`, `- 箇条書き`, `| 表 |`, `![](図)`, `> ノート`）。

構成は毎回同じにする。

1. 今回の結論（3 行以内）
2. 数値（表。出典のドキュメント名を必ず添える）
3. 解釈の限界（床の内外、検出力、seed 数）
4. 次の 2 週間でやること
5. 判断を仰ぎたい点

**すべての数値は結果記録から引く。** 記録に無ければスライドに載せない。
不確かなものは「未確定」と書く。

## 4. .pptx を生成する

訓練環境では動かさない。初回のみ環境を作る。

```
conda create -y -n ard-report python=3.11
conda run -n ard-report pip install -r requirements/reporting.txt
```

生成する。

```
conda run -n ard-report python scripts/ardx/build_slides.py \
  reports/biweekly/<YYYY-MM-DD>.md \
  --output reports/biweekly/<YYYY-MM-DD>.pptx \
  --template reports/biweekly/template.pptx   # あれば
```

出力は通常の .pptx なので、本人が PowerPoint で直接編集できる。
**見た目を変えたくなったらテンプレート側を直す。スクリプトは触らない。**

## 5. 未決事項を報告する

最後に、次を短く列挙して終える。

- 走っている実験と、その完了予定
- 判断待ちの事項（人間が決めるべきもの）
- 着手前提が埋まっていないプラン

## やらないこと

- 結果を見てからの係数・閾値・親の選び直し
- 記録に無い数値のスライドへの記載
- 副次 arm の陽性を結論として報告すること
