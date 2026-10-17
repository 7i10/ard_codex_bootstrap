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
| `<yymmdd>_naito.pptx` | 生成物（発表日、例 261019_naito.pptx。`meta.output` で指定） | commit しない |
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
  進捗は実験のまとまり（Phase など）ごとにセクションを分け、各まとまりの最初に「何をする段階か」「共通の設定」「何を変えたか」を1枚で示す。結果と考察は同じページに書く（考察だけを別セクションに分けない）。
  - 表紙・目次・参考文献は自動で作られる。
  - 先行研究は、進捗で使う論文があるときだけ入れる。最大2本、1本につき2枚：
    - 1枚目：何をした研究か
    - 2枚目：自分の研究にどう使うか、何が違うか
- 本編は約1分に1枚。15分なら本編10〜13枚。入りきらない詳細は付録（セクション「付録」）に回す。
- 進捗の最初に「今回の結論」を1枚置き、要点を3つ以内で示す。最後の「まとめ」で同じ要点を回収する。
- 表紙：`event: 班ゼミ`、日付だけが毎回変わる。
- **発表者ノート**：表紙・目次・参考文献以外の全ページに、話す原稿を `script:` で書く（です・ます調、1枚 300〜600字、約1分）。補足のメモは `notes:` に書く。ノートでは「【メモ】」の見出しの下に入るので、原稿と混ざらない。
- 量が多い回は、各節の頭に `type: agenda`（目次の再掲。今の節を強調）を置いて現在地を示す。
- 1枚1メッセージ。右上の要点には結論の文を書く（例「SiLU で頑健性 +2.6pt」）。「実験条件」のようなラベルは不可。
- 事実・解釈・仮説を書き分ける。効果には揺れの幅（seed）と判定（効果あり／保留／効果なし）を添える。seed 1本の結果はそうと明記する。
- うまくいかなかった結果と、判断を仰ぎたい点も載せる。ゼミの目的はフィードバックをもらうこと。
- 計画は1枚。左に次回まで（ミクロ、日付つき。決まっていない先は「◯日以降は未定」）、右に修論まで（マクロ）。
- まとめの3つ目の欄は「考えるべき点」（「相談したい点」にしない）。

書き方の決まり（2026-10-19 の人間のレビューから）：
- **対象や目的を盛らない**。言っていないこと（例「スマホ向け」）を書かない。定義や閾値（例「10M 未満」）には必ず根拠を添える。根拠のない境界は作らない。
- **聞き手が知らない語は、初出のページで一言説明する**。手法名（AWP、TRADES、IDBH、EMA、SE …）、学習率の名前（「cosine」ではなく「学習率の cosine 減衰」）、活性化関数（SiLU・GELU・hard-swish の違いは式と形で）、DSP、stem、β2 など。比べる手法が多い回は「比べた要素」の表（名前／一言で／種類）を1枚入れる。
- **指示語で済ませない**。「最初の3本」「代わりの案」「DSP に優しい案」などは、中身（名前）を書く。
- **内部の開発用語を避ける**（run id、plan 番号、arm、コミット）。発表者ノートのメモでは可。
- **主張は根拠の強さに合わせる**。まだ確かめていない結論は「（仮説）」「と考える」を付ける。「学習不足」「過学習」と言うときは、学習画像と検証の両方の曲線を見せる。
- **比べる相手を揃える**。表の基準行と、差の計算に使う基準を一致させる。揃わないときは注記する。
- **予想される質問に先に答える**（例：なぜエポックを増やさないのか、学習率を下げたときの急な上昇は何か）。スライドに入らなければ発表者ノートへ。
- **改行は意味の切れ目で入れる**（`\n`）。1行の目安：3列カード約15字、2列カード約23字、右の補足欄約17字。1〜2文字だけ次の行にはみ出す形を残さない。
- **引用は本文の中で** `[@key]` と書く（例「MobileNetV4-S [@mobilenetv4]」）。フッターにはそのページで引用した文献の短い形（「[1] Qin+ 2024」）が自動で出る。
- エポックのように横軸の項目が多い折れ線グラフは、ラベルを間引く（1, 5, 10, … 以外のカテゴリを空白 " " にする。描画ソフトによっては `label_every` が効かず、文字が潰れる）。
- グラフの系列は見分けやすい色にする（活性化関数の比較などは黒・青・緑・赤）。数式は文字で書き、PowerPoint の数式エディタでの清書は人間（PowerPoint の Claude）に任せる。

### 3. `deck.yaml` を書き、生成する

```bash
node tools/slides/build_deck.js reports/biweekly/<日付>/deck.yaml   # → meta.output（例 261019_naito.pptx）
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
conda run -n ard-report python $SK/scripts/office/validate.py <yymmdd>_naito.pptx
conda run -n ard-report markitdown <yymmdd>_naito.pptx | grep -nE "TODO|未確認|xxx"   # 残り物
conda run -n ard-report python $SK/scripts/office/soffice.py --headless --convert-to pdf <yymmdd>_naito.pptx
pdftoppm -jpeg -r 80 <yymmdd>_naito.pdf render/s
```

- 画像は全ページ見る。はみ出し、重なり、改行で切れた数字と単位、空いた枠を確認する。見落としを防ぐため、別のサブエージェント（Sonnet）にも見せる。
- 確認には Linux の LibreOffice を使う。ここには游ゴシックが無いので、改行の位置は Mac の PowerPoint と少しずれる。文字の欄には1割ほど余裕を持たせる。
- 直したらその分だけ描き直して確認し、終わりにする。

### 4.5 渡す前のセルフレビュー（必須）
聞き手（指導教員と研究室の学生）になったつもりで全ページを読み、次を確かめて直す。
- 説明なしの用語、指示語だけの言及、根拠のない定義や閾値が残っていないか（上の「書き方の決まり」）。
- 各ページの右上の要点は、そのページの結論の文になっているか。主張が根拠より強くないか。
- 表の数字と記録（facts.md）が一致しているか。比べる相手が揃っているか。
- 聞かれそうな質問に、スライドかノートで答えているか。
- 全ページに発表原稿があり、15分に収まる量か（本編10〜15枚）。
見つけた点と直した点を、渡すときに短く報告する。

### 5. 人間に渡す
- .pptx の場所と、構成の要約を伝える。仕上げは人間が PowerPoint（Claude in PowerPoint）で行う。
- 枠・タイトル・要点・出典番号・ページ番号はレイアウトの placeholder に入っている。PowerPoint 側で色やフォントを変えても崩れない。
- 最後に、判断待ちの事項と、走っている実験の完了予定を短く列挙する。

## deck.yaml の書式

```yaml
meta:                       # 表紙（毎回変わるのは date と output だけ）
  event: 班ゼミ
  output: 261019_naito.pptx # <yymmdd>_naito.pptx（発表日）
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
    script: 話す原稿（発表者ノート。です・ます調）
    notes: 補足のメモ（ノートでは原稿と区別して表示）
    ...                       # 型ごとの項目（下）
```

| 型 | 項目 |
|---|---|
| bullets | `body: [文, [入れ子の文]]`、`size` |
| cards | `cards: [{head, body: [...]}]`、`cols`、`lead`（上の一文）、`size`、`fill: true`（下まで伸ばす。既定は中身に合わせた高さ） |
| overview | `blocks: [{head, body}]×3`、`timeline: [{label, note, current}]` |
| table | `table: {header, rows, highlight: [[行,列]], col_w: [比率], caption, caption_pos: top, size, row_h}`、`side: [...]`、`side_w`。数字は末尾の0を残すため文字列（"3.70"）で書く。見出しの改行は `\n` |
| chart | `chart: {kind: bar|line, categories, series: [{name, values}], colors, y_title, format, min, max, horizontal, label_every}`、`side` |
| figure | `image`（deck.yaml からの相対パス）、`caption`、`side` |
| plan | `micro: [...]`、`macro: [{period, item, current}]` |
| agenda | 項目なし（その section を強調した目次） |

どの型にも `full: true` を付けると、要点の欄を省いてページ全体を本文に使う（大きな一覧表など）。

文中の `**語**` は太字、`==語==` は強調色（赤）。強調色は1枚に1〜2か所まで。

## やらないこと
- 記録にない数字を載せる。単純な差以外の統計を作る。
- 結果を見てから閾値や判定を変える。副次的な結果を結論として報告する。
- 本文に参考文献の全文を書く（フッターは [n] だけ。全文は参考文献ページ）。
- 生成物（.pptx、.pdf、render/）や、人間から預かった過去の資料を commit する。
