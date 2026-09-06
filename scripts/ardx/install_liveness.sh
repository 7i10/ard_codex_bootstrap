#!/usr/bin/env bash
# Install the cross-host liveness monitor on this host.
#   install_liveness.sh <SELF_NAME> <PEER_NAME> <PEER_HOST> <NTFY_TOPIC>
# Runs entirely in the user session; needs no root, but needs lingering enabled
# (`loginctl enable-linger $USER`) or the timers stop when you log out.
set -euo pipefail
SELF_NAME="$1"; PEER_NAME="$2"; PEER_HOST="$3"; NTFY_TOPIC="$4"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$HOME/.local/bin" "$HOME/.config/ardx" "$HOME/.config/systemd/user"
install -m 0755 "$HERE/liveness.sh" "$HOME/.local/bin/ardx-liveness"
umask 077
cat > "$HOME/.config/ardx/liveness.env" <<CONF
# The ntfy topic name is the only secret here: on the public ntfy.sh server,
# anyone who knows it can read and post.  Keep this file mode 600 and out of git.
SELF_NAME=$SELF_NAME
PEER_NAME=$PEER_NAME
PEER_HOST=$PEER_HOST
NTFY_TOPIC=$NTFY_TOPIC
FAIL_THRESHOLD=3
CONF
umask 022

for u in ardx-liveness.service ardx-liveness.timer ardx-heartbeat.service ardx-heartbeat.timer; do
  install -m 0644 "$HERE/systemd/$u" "$HOME/.config/systemd/user/$u"
done
systemctl --user daemon-reload
systemctl --user enable --now ardx-liveness.timer ardx-heartbeat.timer

linger=$(loginctl show-user "$USER" 2>/dev/null | sed -n 's/^Linger=//p')
echo "installed on $SELF_NAME, watching $PEER_NAME ($PEER_HOST); Linger=${linger:-unknown}"
[ "$linger" = yes ] || echo "WARNING: lingering is off, so these timers stop when you log out. Run: sudo loginctl enable-linger $USER"
systemctl --user list-timers ardx-liveness.timer ardx-heartbeat.timer --no-pager
