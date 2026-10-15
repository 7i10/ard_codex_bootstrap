---
name: biweekly-review
description: Build the biweekly lab-seminar progress deck (.pptx, Japanese, 15 min) from the result records — gather facts with sources, propose an outline for approval, generate the deck with tools/slides/build_deck.js, render and QA it — then hand it to the human for refinement in PowerPoint (Claude in PowerPoint). Use every two weeks before the seminar, or when the user asks for 発表資料 / スライド / ゼミ資料.
---

# 隔週ゼミ発表資料（15分、PowerPoint）

目的は、指導教員と研究室に**進捗を正確に伝え、フィードバックをもらうこと**。進捗を飾らない。
数字はすべて記録から引く。記録にない数字は載せない。不確かなものは「未確定」と書く。

成果物は `reports/biweekly/<YYYY-MM-DD>/`（発表日）に置く。

| ファイル | 中身 | git |
|---|---|---|
| `facts.md` | 今回使う事実と数字。すべてに出典（ファイル:行、記録名） | commit |
| `deck.yaml` | スライドの中身（下の書式） | commit |
| `deck.pptx` | 生成物 | commit しない |
| `render/` | QA 用の画像 | commit しない |

参考文献の共通の台帳は `reports/biweekly/references.yaml`（キー → IEEE 形式の文字列）。

## 手順（前の段が終わるまで次へ進まない）

### 0. 記録の取りこぼしを確認する
`python3 scripts/ardx/status.py` で、終わったのに記録に入っていない run を確認する。
Ferret の run は自動では取り込まれない。終わったものは Hamster の runs/ に rsync して watcher に拾わせる。
取りこぼしは発表前に記録へ入れる（`/experiment-postrun`）。記録に入らない数字はスライドに載せない。

### 1. 事実を集める → `facts.md`
前回の発表以降（初回は対象期間の始まりから）の結果・決定・失敗を、出典つきで集める。
主な出典：
- `docs/plans/0103-discussion-register.md`
- `docs/decisions/*.md`
- 計画書（`docs/plans/*.md`）
- `docs/debugging/*.md`
- run の `epoch-metrics.jsonl`

量が多いときはサブエージェントに任せてよい。ただし数字の写し間違いを防ぐため、出典の行を必ず残させる。

### 2. 構成案を作り、人間の承認をもらう
次の表をチャットで示し、承認をもらってから deck.yaml を書く。

| # | セクション | タイトル | 右上の要点（結論の文） | 型 | 根拠（表・図・数字） | 出典 |
|---|---|---|---|---|---|---|

構成の決まり：
- 流れ：表紙 → 目次 → 研究内容（1枚）→ 先行研究（任意）→ 進捗 → まとめ → 計画（1枚）→ 参考文献。
  - 表紙・目次・参考文献は自動で作られる。
  - 先行研究は、進捗で使う論文があるときだけ入れる。最大2本、1本につき2枚：
    - 1枚目：何をした研究か
    - 2枚目：自分の研究にどう使うか、何が違うか
- 本編は約1分に1枚。15分なら本編10〜13枚。入りきらない詳細は付録（セクション「付録」）に回す。
- 進捗の最初に「今回の結論」を1枚置き、要点を3つ以内で示す。最後の「まとめ」で同じ要点を回収する。
- 量が多い回は、各節の頭に `type: agenda`（目次の再掲。今の節を強調）を置いて現在地を示す。
- 1枚1メッセージ。右上の要点には結論の文を書く（例「SiLU で頑健性 +2.6pt」）。「実験条件」のようなラベルは不可。
- 事実・解釈・仮説を書き分ける。効果には揺れの幅（seed）と判定（効果あり／保留／効果なし）を添える。seed 1本の結果はそうと明記する。
- うまくいかなかった結果と、判断を仰ぎたい点も載せる。ゼミの目的はフィードバックをもらうこと。
- 計画は1枚。左に次回まで（ミクロ）、右に修論まで（マクロ）。

### 3. `deck.yaml` を書き、生成する

```bash
node tools/slides/build_deck.js reports/biweekly/<日付>/deck.yaml   # → deck.pptx
```

初回だけ：
```bash
(cd tools/slides && npm install)
conda create -y -n ard-report python=3.11 && conda run -n ard-report pip install -r requirements/reporting.txt
```

見た目（色・フォント・枠）は `tools/slides/build_deck.js` の THEME とレイアウトで一括管理する。各ページで個別に色や位置を指定しない。

### 4. QA（必須）
```bash
SK=<pptx skill の場所>   # anthropic-skills:pptx の base directory
conda run -n ard-report python $SK/scripts/office/validate.py deck.pptx
conda run -n ard-report markitdown deck.pptx | grep -nE "TODO|未確認|xxx"   # 残り物
conda run -n ard-report python $SK/scripts/office/soffice.py --headless --convert-to pdf deck.pptx
pdftoppm -jpeg -r 80 deck.pdf render/s
```

- 画像は全ページ見る。はみ出し、重なり、改行で切れた数字と単位、空いた枠を確認する。見落としを防ぐため、別のサブエージェント（Sonnet）にも見せる。
- 確認には Linux の LibreOffice を使う。ここには游ゴシックが無いので、改行の位置は Mac の PowerPoint と少しずれる。文字の欄には1割ほど余裕を持たせる。
- 直したらその分だけ描き直して確認し、終わりにする。

### 5. 人間に渡す
- deck.pptx の場所と、構成の要約を伝える。仕上げは人間が PowerPoint（Claude in PowerPoint）で行う。
- 枠・タイトル・要点・出典番号・ページ番号はレイアウトの placeholder に入っている。PowerPoint 側で色やフォントを変えても崩れない。
- 最後に、判断待ちの事項と、走っている実験の完了予定を短く列挙する。

## deck.yaml の書式

```yaml
meta:                       # 表紙（毎回変わるのは date だけ）
  event: 定期ゼミ
  title: <研究題目>
  date: 2026年 10月19日（月）
  affiliation: [早稲田大学　基幹理工学研究科　電子物理システム学専攻, 史研究室　修士2年]
  name: 内藤 舜介
references_file: ../references.yaml   # キー → IEEE 文字列。本文の cite の順に自動で番号を振る
slides:
  - section: 3. 進捗          # 左上の青いタブ。目次はこの値から自動で作る
    type: table               # bullets | cards | overview | table | chart | figure | plan | agenda
    title: Phase 2 の結果一覧  # 短い名詞句
    point: 最適化を助ける工夫は効き、正則化は逆効果   # 右上の要点＝結論の文
    cite: [sat]               # フッターに [n] と出る
    notes: 話す内容（発表者ノート）
    ...                       # 型ごとの項目（下）
```

| 型 | 項目 |
|---|---|
| bullets | `body: [文, [入れ子の文]]`、`size` |
| cards | `cards: [{head, body: [...]}]`、`cols`、`lead`（上の一文） |
| overview | `blocks: [{head, body}]×3`、`timeline: [{label, note, current}]` |
| table | `table: {header, rows, highlight: [[行,列]], col_w: [比率], caption, size}`、`side: [...]`、`side_w` |
| chart | `chart: {kind: bar, categories, series: [{name, values}], y_title, min, max, horizontal}`、`side` |
| figure | `image`（deck.yaml からの相対パス）、`caption`、`side` |
| plan | `micro: [...]`、`macro: [{period, item, current}]` |
| agenda | 項目なし（その section を強調した目次） |

文中の `**語**` は太字、`==語==` は強調色（赤）。強調色は1枚に1〜2か所まで。

## やらないこと
- 記録にない数字を載せる。単純な差以外の統計を作る。
- 結果を見てから閾値や判定を変える。副次的な結果を結論として報告する。
- 本文に参考文献の全文を書く（フッターは [n] だけ。全文は参考文献ページ）。
- 生成物（deck.pptx、render/）や、人間から預かった過去の資料を commit する。
