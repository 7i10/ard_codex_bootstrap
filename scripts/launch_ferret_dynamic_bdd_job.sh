#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/../.agents/skills/run-on-ferret/scripts/ferret-common"
run_id= seed= arm= gpu= source_sha= epochs=115
parent_config= parent_config_sha= parent_checkpoint= parent_checkpoint_sha= preflight=false
while (($#)); do
  case "$1" in
    --run-id) run_id=$2; shift 2;;
    --seed) seed=$2; shift 2;;
    --arm) arm=$2; shift 2;;
    --gpu) gpu=$2; shift 2;;
    --source-sha) source_sha=$2; shift 2;;
    --epochs) epochs=$2; shift 2;;
    --parent-config) parent_config=$2; shift 2;;
    --parent-config-sha) parent_config_sha=$2; shift 2;;
    --parent-checkpoint) parent_checkpoint=$2; shift 2;;
    --parent-checkpoint-sha) parent_checkpoint_sha=$2; shift 2;;
    --preflight) preflight=true; shift;;
    *) die "unknown argument $1";;
  esac
done
[[ $epochs == 115 ]] || die 'dynamic-BDD screen is frozen to runtime epochs=115'
valid_run_id "$run_id"; valid_sha "$source_sha"
[[ $parent_config_sha =~ ^[0-9a-fA-F]{64}$ ]] || die 'parent config SHA-256 must be exactly 64 hex characters'
[[ $parent_checkpoint_sha =~ ^[0-9a-fA-F]{64}$ ]] || die 'parent checkpoint SHA-256 must be exactly 64 hex characters'
[[ $seed == dev-1 || $seed == dev-2 ]] || die 'invalid seed'
[[ $arm == control || $arm == dpm || $arm == dbdd || $arm == sbdd ]] || die 'invalid arm'
[[ $gpu =~ ^[01]$ ]] || die 'Ferret dynamic-BDD jobs may use only GPU 0 or 1'
[[ -f $parent_config && -f $parent_checkpoint ]] || die 'parent config and checkpoint must exist locally'
[[ $(sha256sum "$parent_config" | awk '{print $1}') == "${parent_config_sha,,}" ]] || die 'parent config SHA-256 mismatch'
[[ $(sha256sum "$parent_checkpoint" | awk '{print $1}') == "${parent_checkpoint_sha,,}" ]] || die 'parent checkpoint SHA-256 mismatch'

# A Ferret worktree does not imply that the large, hash-bound continuation
# parents are present there.  Materialize the two exact parent inputs into a
# seed-private cache first, under a local lock shared by concurrent arms.  A
# remote launch is impossible until the remote byte hashes match.
seed_number=${seed#dev-}
remote_parent_root="$FERRET_RUN_ROOT/inputs/ert-i100-s2-dynamic-bdd-v1/$seed"
remote_config="$remote_parent_root/idbh-s100-s$seed_number/resolved_config.yaml"
remote_checkpoint="$remote_parent_root/seed$seed_number/s100/epoch-100.pt"
exec 8>"/tmp/ard-i100-dynamic-bdd-parent-${seed}.lock"
flock -x 8
"${FERRET_SSH[@]}" "mkdir -p $(q "$(dirname "$remote_config")") $(q "$(dirname "$remote_checkpoint")")"
rsync -a --partial "$parent_config" "$FERRET_HOST:$remote_config"
rsync -a --partial "$parent_checkpoint" "$FERRET_HOST:$remote_checkpoint"
"${FERRET_SSH[@]}" "config=$(q "$remote_config") checkpoint=$(q "$remote_checkpoint") expected_config=$(q "${parent_config_sha,,}") expected_checkpoint=$(q "${parent_checkpoint_sha,,}") bash -s" <<'SH'
set -Eeuo pipefail
test -f "$config" && test -f "$checkpoint"
test "$(sha256sum "$config" | awk '{print $1}')" = "$expected_config"
test "$(sha256sum "$checkpoint" | awk '{print $1}')" = "$expected_checkpoint"
SH
if $preflight; then
  printf 'parent preflight passed for %s/%s: %s %s\n' "$seed" "$arm" "$remote_config" "$remote_checkpoint"
  exit 0
fi
remote_output="$(run_dir "$run_id")/outputs"
base_run_id="$run_id"
attempt="${ARD_ORCH_ATTEMPT:-1}"
run_id="${base_run_id}-a${attempt}"
valid_run_id "$run_id"
remote_output="$(run_dir "$run_id")/outputs"
orchestration_env=(
  "ARD_ORCH_CAMPAIGN_ID=${ARD_ORCH_CAMPAIGN_ID:-}"
  "ARD_ORCH_JOB_ID=${ARD_ORCH_JOB_ID:-}"
  "ARD_ORCH_ATTEMPT=${ARD_ORCH_ATTEMPT:-1}"
  "ARD_ORCH_ATTEMPT_ID=${ARD_ORCH_ATTEMPT_ID:-$run_id}"
)
"$(dirname "$0")/../.agents/skills/run-on-ferret/scripts/ferret-prepare" --sha "$source_sha" --run-id "$run_id"
"$(dirname "$0")/../.agents/skills/run-on-ferret/scripts/ferret-launch" --run-id "$run_id" --gpus "$gpu" --launcher direct -- \
  /usr/bin/env PYTHONPATH=src ARD_STAGEWISE_RUN_ROOT="$remote_parent_root" "${orchestration_env[@]}" \
  "$FERRET_PYTHON" scripts/run_ert_i100_s2_dynamic_bdd_job.py --seed "$seed" --arm "$arm" --output "$remote_output" --device cuda --epochs "$epochs" --expected-source-sha "$source_sha"
