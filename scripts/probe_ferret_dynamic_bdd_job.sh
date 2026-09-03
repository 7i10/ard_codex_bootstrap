#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/../.agents/skills/run-on-ferret/scripts/ferret-common"
run_id= local_output=
while (($#)); do
  case "$1" in
    --run-id) run_id=$2; shift 2;;
    --local-output) local_output=$2; shift 2;;
    *) die "unknown argument $1";;
  esac
done
valid_run_id "$run_id"
attempt="${ARD_ORCH_ATTEMPT:-1}"
run_id="${run_id}-a${attempt}"
valid_run_id "$run_id"
[[ $local_output == /* ]] || die 'local output must be absolute'
run=$(run_dir "$run_id"); dest="$HAMSTER_RESULT_ROOT/$run_id"
status_json=$("$(dirname "$0")/../.agents/skills/run-on-ferret/scripts/ferret-status" --run-id "$run_id")
status=$(printf '%s' "$status_json" | "$FERRET_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["status"])')
[[ $status == completed ]] || exit 1
if [[ ! -f "$dest/.collected" ]]; then
  "$(dirname "$0")/../.agents/skills/run-on-ferret/scripts/ferret-collect" --run-id "$run_id"
  mkdir -p "$dest"
  rsync -a --partial --safe-links \
    --include='/outputs/' --include='/outputs/**/' --include='/outputs/**/*.parquet' \
    --exclude='*' "$FERRET_HOST:$run/" "$dest/"
  printf '%s\n' "$run_id" > "$dest/.collected"
fi
mkdir -p "$local_output"
rsync -a --partial --safe-links "$dest/" "$local_output/"
test -f "$local_output/outputs/production-summary.json" || {
  echo "remote job completed but expected production summary was not collected" >&2
  exit 2
}
