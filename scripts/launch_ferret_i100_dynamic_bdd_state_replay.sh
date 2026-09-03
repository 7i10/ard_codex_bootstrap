#!/usr/bin/env bash
set -Eeuo pipefail

# Launch a hash-bound, checkpoint-only state replay on Ferret.  This is an
# execution wrapper; it never trains or mutates a checkpoint.
source "$(dirname "$0")/../.agents/skills/run-on-ferret/scripts/ferret-common"

run_id= gpu= source_sha= remote_config= remote_config_sha= teacher_sha=
declare -a checkpoints=() checkpoint_hashes=()
while (($#)); do
  case "$1" in
    --run-id) run_id=$2; shift 2;;
    --gpu) gpu=$2; shift 2;;
    --source-sha) source_sha=$2; shift 2;;
    --remote-config) remote_config=$2; shift 2;;
    --remote-config-sha256) remote_config_sha=$2; shift 2;;
    --checkpoint) checkpoints+=("$2"); shift 2;;
    --checkpoint-sha256) checkpoint_hashes+=("$2"); shift 2;;
    --teacher-sha256) teacher_sha=$2; shift 2;;
    *) die "unknown argument $1";;
  esac
done
valid_run_id "$run_id"; valid_sha "$source_sha"
[[ $gpu =~ ^[01]$ ]] || die 'state replay may use Ferret GPU 0 or 1 only'
[[ $remote_config == /* && $remote_config_sha =~ ^[0-9a-f]{64}$ && $teacher_sha =~ ^[0-9a-f]{64}$ ]] || die 'invalid config or Teacher identity'
((${#checkpoints[@]})) || die 'at least one checkpoint is required'
((${#checkpoints[@]} == ${#checkpoint_hashes[@]})) || die 'checkpoint and hash counts differ'

declare -A expected=()
for index in "${!checkpoints[@]}"; do
  epoch=${checkpoints[$index]%%=*}; path=${checkpoints[$index]#*=}; digest=${checkpoint_hashes[$index]#*=}
  [[ $epoch =~ ^[0-9]+$ && $path == /* && $digest =~ ^[0-9a-f]{64}$ ]] || die 'checkpoint formats are EPOCH=/absolute/path and EPOCH=<sha256>'
  [[ -z ${expected[$epoch]:-} ]] || die "duplicate checkpoint epoch: $epoch"
  expected[$epoch]=$digest
done

remote_checkpoints=()
for item in "${checkpoints[@]}"; do remote_checkpoints+=("${item#*=}"); done
"${FERRET_SSH[@]}" "config=$(q "$remote_config") config_sha=$(q "$remote_config_sha") teacher_sha=$(q "$teacher_sha") checkpoints=$(q "${remote_checkpoints[*]}") hashes=$(q "${checkpoint_hashes[*]}") teacher=$(q "$FERRET_REPO_ROOT/teacher_cache/robustbench/Chen2021LTD_WRN34_10.pt") bash -s" <<'SH'
set -Eeuo pipefail
test -f "$config" && test -f "$teacher"
test "$(sha256sum "$config" | awk '{print $1}')" = "$config_sha"
test "$(sha256sum "$teacher" | awk '{print $1}')" = "$teacher_sha"
read -r -a paths <<<"$checkpoints"; read -r -a expected <<<"$hashes"
for index in "${!paths[@]}"; do
  epoch=${expected[$index]%%=*}; digest=${expected[$index]#*=}
  test -f "${paths[$index]}"
  test "$(sha256sum "${paths[$index]}" | awk '{print $1}')" = "$digest"
  test "$epoch" = "$(basename "${paths[$index]}" | sed -E 's/epoch-([0-9]+)\.pt/\1/')"
done
SH

attempt="${ARD_ORCH_ATTEMPT:-1}"
attempt_run_id="${run_id}-a${attempt}"
valid_run_id "$attempt_run_id"
"$(dirname "$0")/../.agents/skills/run-on-ferret/scripts/ferret-prepare" --sha "$source_sha" --run-id "$attempt_run_id"

arguments=(
  /usr/bin/env PYTHONPATH=src ARD_NUM_WORKERS=4
  "${FERRET_PYTHON}" scripts/replay_ert_i100_s2_dynamic_bdd_states.py
  --config "$remote_config"
  --output "$(run_dir "$attempt_run_id")/outputs/state-replay"
  --expected-source-sha "$source_sha"
  --expected-teacher-sha256 "$teacher_sha"
  --device cuda
)
for index in "${!checkpoints[@]}"; do
  epoch=${checkpoints[$index]%%=*}
  arguments+=(--checkpoint "${checkpoints[$index]}" --checkpoint-sha256 "${checkpoint_hashes[$index]}")
done
"$(dirname "$0")/../.agents/skills/run-on-ferret/scripts/ferret-launch" --run-id "$attempt_run_id" --gpus "$gpu" --launcher direct -- "${arguments[@]}"
