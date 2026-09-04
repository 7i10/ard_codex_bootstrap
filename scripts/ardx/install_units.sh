#!/usr/bin/env bash
# Install/remove the ardx-watch user unit.  Requires linger (docs/debugging/0024-systemd-user-logout-killed-saad-oracles.md).
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_SRC="$HERE/systemd/ardx-watch.service"
UNIT_NAME="ardx-watch.service"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_DST="$UNIT_DIR/$UNIT_NAME"
RUNTIME_ROOT="${ARDX_RUNTIME_ROOT:-/home/islab/workspace-local/shunsuke.naito/ard-runtime/ard_codex_bootstrap}"
ARDX_DIR="$RUNTIME_ROOT/orchestration/ardx"

usage() { echo "usage: install_units.sh [--dry-run|--install|--uninstall]"; }

linger_state() { loginctl show-user "$(id -un)" --property=Linger 2>/dev/null | cut -d= -f2; }

require_linger() {
  local state; state="$(linger_state || true)"
  if [ "$state" != "yes" ]; then
    echo "refusing to install: loginctl Linger=${state:-unknown} for $(id -un)." >&2
    echo "Without linger the user manager dies at logout and the watcher dies with it;" >&2
    echo "see docs/debugging/0024-systemd-user-logout-killed-saad-oracles.md. Run: sudo loginctl enable-linger $(id -un)" >&2
    exit 3
  fi
}

MODE="${1:---dry-run}"
case "$MODE" in
  --dry-run)
    echo "unit source:      $UNIT_SRC"
    echo "unit destination: $UNIT_DST"
    echo "runtime state:    $ARDX_DIR"
    echo "linger:           $(linger_state || echo unknown)"
    [ -f "$UNIT_SRC" ] || { echo "missing unit file: $UNIT_SRC" >&2; exit 1; }
    echo "would run: install -Dm644 $UNIT_SRC $UNIT_DST"
    echo "would run: systemctl --user daemon-reload"
    echo "would run: systemctl --user enable --now $UNIT_NAME"
    [ "$(linger_state || true)" = "yes" ] || echo "warning: linger is not enabled; --install would abort (docs/debugging/0024)."
    ;;
  --install)
    require_linger
    mkdir -p "$ARDX_DIR/claude-runs"
    install -Dm644 "$UNIT_SRC" "$UNIT_DST"
    systemctl --user daemon-reload
    systemctl --user enable --now "$UNIT_NAME"
    systemctl --user --no-pager status "$UNIT_NAME" || true
    ;;
  --uninstall)
    systemctl --user disable --now "$UNIT_NAME" || true
    rm -f "$UNIT_DST"
    systemctl --user daemon-reload
    echo "removed $UNIT_DST"
    ;;
  -h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
