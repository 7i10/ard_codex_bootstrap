#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/../.agents/skills/run-on-ferret/scripts/ferret-common"
run_id= seed= arm= gpu= source_sha= epochs=115
while (($#)); do
  case "$1" in
    --run-id) run_id=$2; shift 2;;
    --seed) seed=$2; shift 2;;
    --arm) arm=$2; shift 2;;
    --gpu) gpu=$2; shift 2;;
    --source-sha) source_sha=$2; shift 2;;
    --epochs) epochs=$2; shift 2;;
    *) die "unknown argument $1";;
  esac
done
[[ $epochs == 115 ]] || die 'dynamic-BDD screen is frozen to runtime epochs=115'
valid_run_id "$run_id"; valid_sha "$source_sha"
[[ $seed == dev-1 || $seed == dev-2 ]] || die 'invalid seed'
[[ $arm == control || $arm == dpm || $arm == dbdd || $arm == sbdd ]] || die 'invalid arm'
[[ $gpu =~ ^[01]$ ]] || die 'Ferret dynamic-BDD jobs may use only GPU 0 or 1'
remote_output="$(run_dir "$run_id")/outputs"
base_run_id="$run_id"
attempt="${ARD_ORCH_ATTEMPT:-1}"
run_id="${base_run_id}-a${attempt}"
valid_run_id "$run_id"
remote_output="$(run_dir "$run_id")/outputs"
"$(dirname "$0")/../.agents/skills/run-on-ferret/scripts/ferret-prepare" --sha "$source_sha" --run-id "$run_id"
"$(dirname "$0")/../.agents/skills/run-on-ferret/scripts/ferret-launch" --run-id "$run_id" --gpus "$gpu" --launcher direct -- \
  /usr/bin/env PYTHONPATH=src ARD_STAGEWISE_RUN_ROOT=/home/shunsukenaito/workspace-local/ard-runs/ard_codex_bootstrap/ert-rslad-stagewise-v1 \
  "$FERRET_PYTHON" scripts/run_ert_i100_s2_dynamic_bdd_job.py --seed "$seed" --arm "$arm" --output "$remote_output" --device cuda --epochs "$epochs"
