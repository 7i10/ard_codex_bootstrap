# Upstream baseline management

## 1. 方針

公式SAAD repositoryは `.external/saad` へcloneし、Git submoduleとして自動的に公開repoへ含めるのではなく、`external.lock.yaml` とbootstrap scriptで固定します。この方式により、upstream sourceを自repoへコピーせず、複数サーバーで同一commitを再取得できます。

固定remote:

```text
https://github.com/HongsinLee/saad.git
```

`external.lock.yaml` は次を固定しています。

```text
remote: https://github.com/HongsinLee/saad.git
commit: 295121c5d2eed827b5b2d6aa42307de809bdfada
license_status: absent
```

bootstrap時にorigin、detached HEAD、clean statusを照合します。branch HEADを追跡しません。

## 2. Lock schema

例:

```yaml
version: 1
repositories:
  saad:
    url: https://github.com/HongsinLee/saad.git
    commit: <40-char-sha>
    fetched_at: <ISO-8601>
    license_file: <path-or-null>
    license_status: verified | absent | unclear
```

## 3. Bootstrap behavior

`scripts/bootstrap_external.py` は:

- `.external`を作成
- lockがある場合はexact commitをcheckout
- lockがない初回だけremote HEADを取得し、commitをlock候補として出力
- remote mismatchをエラー化
- dirty working treeを上書きしない
- clone失敗・checkout失敗を成功扱いしない
- lock更新は明示flagでのみ許可

## 4. License handling

- LICENSE/COPYING等を実際に確認する。
- 見つからない場合は「ライセンス不明」と記録する。
- ライセンス不明コードを自repoへコピー・改変・再配布しない。
- 研究再現用のlocal cloneとして利用し、自作実装は論文記述と観測可能なbehaviorに基づくclean-room構造を優先する。
- 必要なら著者へ許可を確認する。

## 5. Upstream usage

許可する用途:

- CLI/config/defaultの調査
- fixed-batch differential regression
- upstream reproduction launcher
- checkpoint formatの読込adapter
- 実験結果の比較

避ける用途:

- production runtimeからの暗黙import
- sourceの無断copy-paste
- upstream directoryへの未記録の直接編集
- branch HEADを自動追随

## 6. Patches

再現に環境修正が必要な場合:

```text
patches/saad/<short-description>.patch
```

として保存し、元commit、理由、科学的挙動への影響、適用commandを記録します。patch適用後のtree hashもmanifestへ保存します。

## 7. Reproduction status

M2 inspected the pinned local clone at
`295121c5d2eed827b5b2d6aa42307de809bdfada`. No root `LICENSE` or
`COPYING` file was found; the source remains a local oracle only and is not
copied into this repository.

The clean-room M2 baseline path supports PGD-AT, TRADES, RSLAD, and the
approved entropy-only RSLAD ablation, plus the student-only and joint-risk
ablations. The entropy ablation uses Shannon
entropy (`gamma=1`) from the frozen teacher on the student-crafted adversarial
input and multiplies the complete unreduced RSLAD sample loss by
`5 * (H_i - min_valid_global_batch(H))`. Five is an exact method constant, not
a configurable scale. It intentionally has no clipping, mean
preservation, or hard-label fallback.

Known upstream observations, not copied behavior:

- `rslad.py` calls its inner function with `step_size=8/255` and
  `epsilon=2/255`, even though the function signature names those arguments in
  the usual order. M2 keeps the documented repository threat model of radius
  `8/255` and step size `2/255`; this inversion is recorded rather than
  reproduced silently.
- Upstream invokes AutoAttack in the training process after its final save and
  does not expose a separate saved-checkpoint evaluation/resume lifecycle or
  distinct best checkpoint. This repository does not use that behavior; full
  SAAD is only available through `scripts/run_saad_upstream.py` as a verified
  clone subprocess (`--dry-run` by default for inspection).
- This repository does not copy an upstream optimizer/scheduler/training
  schedule into the CIFAR templates. Those values are required explicitly as
  environment-expanded inputs until a dependency-complete T4 reproduction
  validates them. A schema-valid template is not an upstream-parity claim.
- This repository writes and evaluates distinct `best.pt` and `last.pt`, and
  supports epoch-boundary resume. Those lifecycle guarantees are local
  improvements and must not be attributed to the upstream implementation.

Regression coverage includes fixed-batch KL direction/temperature/`T^2`,
unreduced M2 objective terms, entropy weights, frozen teacher parameters, one
optimizer update, and an opt-in subprocess differential. The latter requires
`ARD_RUN_SAAD_ORACLE=1` and a dependency-complete local clone; it skips when
the local oracle cannot import its own optional dependencies.

## 8. Verified versus deferred

Verified during bootstrap:

- lock parsing and exact SHA/origin/clean checkout checks
- absence of a root `LICENSE`/`COPYING` file at the pinned commit
- clean-room fixed-batch formulas and an opt-in subprocess oracle boundary

Deferred and therefore not claimed:

- dependency-complete full SAAD training or evaluation
- CIFAR public-number reproduction
- full AutoAttack comparison
- legal permission to copy, modify, or redistribute upstream source

Because the license file is absent, `.external/saad` remains a local read-only
oracle. No upstream source is vendored into production modules.
