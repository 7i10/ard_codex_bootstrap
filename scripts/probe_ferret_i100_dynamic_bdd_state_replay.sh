#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/../.agents/skills/run-on-ferret/scripts/ferret-common"
run_id= local_output= expected_epochs=
while (($#)); do
  case "$1" in
    --run-id) run_id=$2; shift 2;;
    --local-output) local_output=$2; shift 2;;
    --expected-epochs) expected_epochs=$2; shift 2;;
    *) die "unknown argument $1";;
  esac
done
valid_run_id "$run_id"
[[ $local_output == /* && $expected_epochs =~ ^[0-9]+(,[0-9]+)*$ ]] || die 'invalid local output or expected epochs'
attempt="${ARD_ORCH_ATTEMPT:-1}"
attempt_run_id="${run_id}-a${attempt}"
valid_run_id "$attempt_run_id"
status_json=$("$(dirname "$0")/../.agents/skills/run-on-ferret/scripts/ferret-status" --run-id "$attempt_run_id")
status=$(printf '%s' "$status_json" | "$FERRET_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["status"])')
[[ $status == completed ]] || exit 1
"$(dirname "$0")/../.agents/skills/run-on-ferret/scripts/ferret-collect" --run-id "$attempt_run_id"
rsync -a --partial --safe-links "$HAMSTER_RESULT_ROOT/$attempt_run_id/" "$local_output/"
IFS=, read -r -a epochs <<<"$expected_epochs"
for epoch in "${epochs[@]}"; do
  test -f "$local_output/outputs/state-replay/e${epoch}/state-replay.json"
  test -f "$local_output/outputs/state-replay/e${epoch}/state-rows.parquet"
done
