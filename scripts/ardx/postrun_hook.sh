#!/usr/bin/env bash
# ardx postrun hook: watcher terminal event -> headless `claude -p /experiment-postrun`.
# Usage: postrun_hook.sh KIND ID STATUS      (ARDX_EVENT_JSON carries the full event)
# Always exits 0: a broken hook must never take the watcher down.
set -uo pipefail

log() { printf '%s postrun_hook: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

if [ "$#" -lt 3 ]; then
  log "usage: postrun_hook.sh KIND ID STATUS"
  exit 0
fi
KIND="$1"; ID="$2"; STATUS="$3"

REPO_ROOT="${ARDX_REPO_ROOT:-/home/islab/workspace-local/shunsuke.naito/ard_codex_bootstrap}"
RUNTIME_ROOT="${ARDX_RUNTIME_ROOT:-/home/islab/workspace-local/shunsuke.naito/ard-runtime/ard_codex_bootstrap}"
LOCK_DIR="$RUNTIME_ROOT/locks"
ARDX_DIR="$RUNTIME_ROOT/orchestration/ardx"
RUN_DIR="$ARDX_DIR/claude-runs"
MODEL="${ARDX_MODEL:-opus}"
MAX_TURNS="${ARDX_MAX_TURNS:-60}"

resolve_claude() {
  if [ -n "${ARDX_CLAUDE_BIN:-}" ] && [ -x "${ARDX_CLAUDE_BIN}" ]; then printf '%s\n' "$ARDX_CLAUDE_BIN"; return 0; fi
  local candidate
  candidate="$(ls -1d "$HOME"/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude 2>/dev/null | sort -V | tail -1)"
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then printf '%s\n' "$candidate"; return 0; fi
  if [ -x "$HOME/.local/bin/claude" ]; then printf '%s\n' "$HOME/.local/bin/claude"; return 0; fi
  command -v claude 2>/dev/null && return 0
  return 1
}

# Tool allowlist: mirror .claude/settings.json when it exists, else the literal
# copy of its `permissions.allow` below.  Keep the two in the same spelling --
# Claude Code matches an allowlist entry against the literal command text, so a
# drifted fallback such as `Bash(python3 scripts/ardx/:*)` (a `:*` after a
# slash asks for a space there) silently matches nothing.  `--permission-mode
# acceptEdits` alone cannot run Bash in headless mode (anything that would
# prompt is denied), so the allowlist is what actually lets the postrun verify,
# aggregate and commit.
DEFAULT_TOOLS=(
  Read
  Edit
  Write
  Glob
  Grep
  'Bash(/home/shunsukenaito/.conda/envs/adv/bin/python -m pytest *)'
  'Bash(/home/shunsukenaito/.conda/envs/adv/bin/python -m ard.cli.status *)'
  'Bash(/home/shunsukenaito/.conda/envs/adv/bin/python scripts/verify.py *)'
  'Bash(/home/shunsukenaito/.conda/envs/adv/bin/python scripts/aggregate_*)'
  'Bash(PYTHONPATH=src /home/shunsukenaito/.conda/envs/adv/bin/python -m pytest *)'
  'Bash(PYTHONPATH=src /home/shunsukenaito/.conda/envs/adv/bin/python -m ard.cli.status *)'
  'Bash(PYTHONPATH=src /home/shunsukenaito/.conda/envs/adv/bin/python scripts/verify.py *)'
  'Bash(PYTHONPATH=src /home/shunsukenaito/.conda/envs/adv/bin/python scripts/aggregate_*)'
  'Bash(/home/shunsukenaito/.conda/envs/adv/bin/python scripts/ardx/*)'
  'Bash(python3 scripts/ardx/*)'
  'Bash(/usr/bin/python3 scripts/ardx/*)'
  'Bash(git status *)'
  'Bash(git diff *)'
  'Bash(git log *)'
  'Bash(git show *)'
  'Bash(git branch *)'
  'Bash(git worktree list *)'
  'Bash(git rev-parse *)'
  'Bash(git merge-base *)'
  'Bash(git add *)'
  'Bash(git commit *)'
  'Bash(sha256sum *)'
  'Bash(ls *)'
  'Bash(cat *)'
  'Bash(head *)'
  'Bash(tail *)'
  'Bash(wc *)'
  'Bash(find *)'
  'Bash(grep *)'
  'Bash(rg *)'
  'Bash(nvidia-smi *)'
  'Bash(ssh Ferret nvidia-smi *)'
  'Bash(ssh Ferret uptime *)'
  'Bash(systemctl --user status *)'
  'Bash(systemctl --user is-active *)'
  'Bash(systemctl --user list-units *)'
  'Bash(loginctl show-user *)'
  'Bash(timeout 5 notify-send *)'
  'Bash(notify-send *)'
)

default_tools() {
  local IFS=,
  printf '%s' "${DEFAULT_TOOLS[*]}"
}

allowed_tools() {
  local settings="$REPO_ROOT/.claude/settings.json"
  if [ -f "$settings" ]; then
    local from_settings
    from_settings="$(/usr/bin/python3 - "$settings" <<'PY' 2>/dev/null
import json, sys
try:
    entries = json.load(open(sys.argv[1]))["permissions"]["allow"]
except Exception:
    entries = []
entries = [str(e) for e in entries if isinstance(e, str)]
print(",".join(entries))
PY
)"
    if [ -n "$from_settings" ]; then printf '%s' "$from_settings"; return 0; fi
  fi
  default_tools
}

CLAUDE_BIN="$(resolve_claude || true)"
if [ -z "$CLAUDE_BIN" ]; then
  log "no claude binary found; nothing to do for $KIND $ID ($STATUS)"
  exit 0
fi

TOOLS="$(allowed_tools)"

# The campaign id is not unique across attempt directories (three run dirs share
# `ert-i100-online-state-s2-v1`), so pass the event's own state/manifest path and
# key the lock and log on the run directory as well as the id.
EVENT_META="$(/usr/bin/python3 - <<'PYX' 2>/dev/null
import json, os
try:
    evt = json.loads(os.environ.get("ARDX_EVENT_JSON") or "{}")
    if not isinstance(evt, dict):
        evt = {}
except Exception:
    evt = {}
path = str(evt.get("path") or "")
detail = evt.get("detail")
run_dir = str(detail.get("run_dir") or "") if isinstance(detail, dict) else ""
if not run_dir and path:
    parts = os.path.dirname(path).split(os.sep)
    while parts and parts[-1] in {"run-bundle", "orchestration"}:
        parts.pop()
    run_dir = os.sep.join(parts)
print(path)
print(os.path.basename(run_dir.rstrip(os.sep)))
PYX
)"
STATE_PATH="$(printf '%s\n' "$EVENT_META" | sed -n '1p')"
RUN_SLUG="$(printf '%s\n' "$EVENT_META" | sed -n '2p')"

LOCK_KEY="$ID"
if [ -n "$RUN_SLUG" ] && [ "$RUN_SLUG" != "$ID" ]; then LOCK_KEY="$ID@$RUN_SLUG"; fi
SAFE_ID="$(printf '%s' "$LOCK_KEY" | tr -c 'A-Za-z0-9._-' '_')"
LOCK_FILE="$LOCK_DIR/ardx-postrun-$SAFE_ID.lock"

PROMPT="/experiment-postrun $ID --status $STATUS --kind $KIND"
if [ -n "$STATE_PATH" ]; then PROMPT="$PROMPT --state-path $STATE_PATH"; fi
CMD=("$CLAUDE_BIN" -p "$PROMPT" --model "$MODEL" --max-turns "$MAX_TURNS" --output-format json
     --permission-mode acceptEdits --allowedTools "$TOOLS")

if [ "${ARDX_DRY_RUN:-0}" = "1" ]; then
  printf 'ARDX_DRY_RUN cwd=%s\n' "$REPO_ROOT"
  printf 'ARDX_DRY_RUN log=%s\n' "$RUN_DIR/<utc>-$SAFE_ID.json"
  printf 'ARDX_DRY_RUN lock=%s\n' "$LOCK_FILE"
  printf 'ARDX_DRY_RUN prompt=%s\n' "$PROMPT"
  printf 'ARDX_DRY_RUN cmd='
  printf '%s ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$LOCK_DIR" "$RUN_DIR" || true
exec 9>"$LOCK_FILE" || { log "cannot open lock $LOCK_FILE"; exit 0; }
if ! flock -n 9; then
  log "postrun for $ID already running (lock $LOCK_FILE held); skipping"
  exit 0
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$RUN_DIR/$STAMP-$SAFE_ID.json"
ERR="$RUN_DIR/$STAMP-$SAFE_ID.log"
cd "$REPO_ROOT" || { log "cannot cd $REPO_ROOT"; exit 0; }
log "running postrun for $KIND $ID ($STATUS) -> $OUT"
CLAUDE_CODE_EFFORT_LEVEL=high "${CMD[@]}" >"$OUT" 2>"$ERR" || log "claude exited non-zero; see $ERR"

SUMMARY="$(/usr/bin/python3 - "$OUT" <<'PY' 2>/dev/null
import json, sys
try:
    payload = json.load(open(sys.argv[1]))
except Exception:
    print("no parsable result")
    raise SystemExit(0)
text = payload.get("result") if isinstance(payload, dict) else None
print(" ".join(str(text or payload).split())[:280])
PY
)"
[ -n "$SUMMARY" ] || SUMMARY="no result recorded"
log "postrun for $ID finished: $SUMMARY"
# notify-send blocks ~60-75 s on this host (no D-Bus notification owner, so each
# call waits out activation) while the per-ID flock is still held: bound it.
if command -v notify-send >/dev/null 2>&1; then
  timeout 5 notify-send "ARD postrun $ID" "$SUMMARY" >/dev/null 2>&1 || true
fi
exit 0
