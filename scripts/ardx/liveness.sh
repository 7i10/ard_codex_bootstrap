#!/usr/bin/env bash
# Cross-host liveness monitor.  Each host watches the other and reports to ntfy.
#
# Why TCP port 22 and not ssh or ping:
#   The 2026-09-04 outage had the machine fully alive -- ping answered, rpcbind
#   answered -- while sshd had been removed by an unattended upgrade and never
#   restarted.  Anything that only pings would have reported "fine" for 60 hours.
#   Anything that ssh'es in would need a key on both hosts.  A plain TCP connect
#   to port 22 needs no credentials and distinguishes the two cases directly:
#     ping ok  + port refused  ->  the OS is up, sshd is gone   (09-04)
#     ping fail                ->  the machine or network is down
#
# It alerts only after FAIL_THRESHOLD consecutive failures so that a reboot or a
# brief network blip does not page anyone, and it alerts once on recovery.
#
# Uses no Claude tokens: shell, systemd and curl only.
set -uo pipefail

CONF="${ARDX_LIVENESS_CONF:-$HOME/.config/ardx/liveness.env}"
[ -r "$CONF" ] || { echo "missing config: $CONF" >&2; exit 78; }
# shellcheck disable=SC1090
. "$CONF"
: "${PEER_NAME:?}" "${PEER_HOST:?}" "${SELF_NAME:?}" "${NTFY_TOPIC:?}"
FAIL_THRESHOLD="${FAIL_THRESHOLD:-3}"
STATE="${STATE:-$HOME/.local/state/ardx/liveness-$PEER_NAME}"
mkdir -p "$(dirname "$STATE")"

notify() {  # notify <priority> <tags> <title> <body>
  # A monitor that cannot report is worse than none, so a failed send is logged
  # to the journal rather than swallowed.  Only the host name and its state ever
  # leave this machine: on the public ntfy.sh server the topic name is the only
  # access control, so nothing about the research goes into these messages.
  local err
  if ! err=$(curl -fsS --max-time 20 \
      -H "Title: $3" -H "Priority: $1" -H "Tags: $2" \
      -d "$4" "https://ntfy.sh/$NTFY_TOPIC" 2>&1); then
    echo "$(date -Is) NOTIFY FAILED (${err:-no detail}); wanted to say: $3 -- $4" >&2
    return 1
  fi
}

case "${1:-check}" in
check)
  ping -c 2 -W 3 "$PEER_HOST" >/dev/null 2>&1 && ping_ok=yes || ping_ok=no
  # bash's own /dev/tcp needs no nc installed
  timeout 8 bash -c "exec 3<>/dev/tcp/$PEER_HOST/22" 2>/dev/null && ssh_ok=yes || ssh_ok=no

  if [ "$ssh_ok" = yes ]; then
    fails=0
  else
    fails=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
  fi
  echo "$fails" > "$STATE"

  if [ "$fails" -eq "$FAIL_THRESHOLD" ]; then
    if [ "$ping_ok" = yes ]; then
      notify urgent rotating_light "$PEER_NAME: sshd down" \
        "$PEER_NAME answers ping but refuses port 22. The machine is up and sshd is gone -- the 2026-09-04 pattern. Seen from $SELF_NAME."
    else
      notify urgent rotating_light "$PEER_NAME: unreachable" \
        "$PEER_NAME answers neither ping nor port 22. Machine or network down. Seen from $SELF_NAME."
    fi
  elif [ "$fails" -eq 0 ] && [ -f "$STATE.alerted" ]; then
    notify default white_check_mark "$PEER_NAME: back" "$PEER_NAME answers port 22 again. Seen from $SELF_NAME."
    rm -f "$STATE.alerted"
  fi
  [ "$fails" -ge "$FAIL_THRESHOLD" ] && touch "$STATE.alerted"
  echo "$(date -Is) peer=$PEER_NAME ping=$ping_ok ssh=$ssh_ok consecutive_failures=$fails"
  ;;

heartbeat)
  # Daily proof of life.  Its ABSENCE is the signal that covers the one blind
  # spot of mutual monitoring: if both hosts die at once, neither can alert.
  gpu=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ' ')
  disk=$(df -h "$HOME" | awk 'NR==2{print $5" used"}')
  notify low heartbeat "$SELF_NAME: alive" "GPU ${gpu:-n/a}| home $disk | up $(uptime -p)"
  echo "$(date -Is) heartbeat sent from $SELF_NAME"
  ;;

test)
  notify default bell "$SELF_NAME: test" "Liveness monitoring is wired up on $SELF_NAME, watching $PEER_NAME."
  echo "test notification sent to topic $NTFY_TOPIC"
  ;;

*) echo "usage: $0 {check|heartbeat|test}" >&2; exit 2 ;;
esac
