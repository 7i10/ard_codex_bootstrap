# Hamster SSH lockout: an openssh upgrade stopped sshd and never restarted it

## Failure signature

From **2026-09-04 07:07:44 JST** until the reboot at **2026-09-06 19:07:38 JST** —
60 hours — nothing was listening on Hamster's TCP port 22.  ICMP answered and
rpcbind (111) answered because those daemons were untouched; SSH gave
"connection refused" because there was no sshd.  The host never crashed, never
ran out of anything, and never lost its network.

The trigger was `unattended-upgrade` installing `openssh-server`
`1:9.6p1-3ubuntu13.18` to `13.19`.  The package's own `preinst` stopped
`ssh.service`, and the `postinst` then failed to start it again because of how
this host has `ssh.socket` masked.

Nothing in the ARD project caused this or made it worse.

## Timeline (JST)

| When | What | Status |
|---|---|---|
| 2026-04-04 08:24:44 | `ssh.socket` masked on this host (`/etc/systemd/system/ssh.socket` to `/dev/null`) | VERIFIED (`stat`) |
| 2026-09-04 05:46 / 06:13 | Last successful inbound SSH logins, from Ferret | VERIFIED (`last`, sshd journal) |
| 2026-09-04 06:57:12 | `apt-daily-upgrade.service` starts | VERIFIED (journal) |
| **2026-09-04 07:07:44** | `Stopping ssh.service`; `sshd[1751]: Received signal 15; terminating`; `Stopped ssh.service` | VERIFIED (journal) |
| 2026-09-04 07:07:45-46 | `configure openssh-server 13.19`, `status installed` | VERIFIED (`/var/log/dpkg.log`) |
| 2026-09-04 07:07:47 | apt transaction ends.  **No `Starting ssh.service` ever appears again in this boot.** | VERIFIED (journal) |
| 2026-09-04 13:47 | The one still-open SSH session (opened 05:43) closes.  The last way in is gone. | VERIFIED (`last`) |
| 2026-09-05 17:03 / 17:06 | Hamster to Ferret SSH still works; outbound uses `openssh-client`, unaffected | consistent with the above |
| 2026-09-05 18:01:53-18:04:08 | Two ARD prefix and freeze jobs run and finish `rc=0` | VERIFIED (run logs) |
| 2026-09-06 06:18-06:20 | `unattended-upgrade` installs kernel 7.0.0-31 and nvidia 595.  No automatic reboot. | VERIFIED (`/var/log/apt/history.log`) |
| 2026-09-06 19:00-19:01 | USB keyboard and mouse plugged in at the console; `fbcon: Taking over console` | VERIFIED (journal) |
| 2026-09-06 19:04:40-19:05:51 | `sysrq: This sysrq operation is disabled.` four times, then `sysrq: Emergency Sync`.  Journal ends. | VERIFIED (journal) |
| 2026-09-06 19:07:32 | New boot, kernel 7.0.0-31, `ssh.service` active | VERIFIED (`systemctl status ssh`) |

## Root cause

Ubuntu's `openssh-server` maintainer scripts do this on upgrade.

`/var/lib/dpkg/info/openssh-server.preinst`:

```sh
if [ -z "${DPKG_ROOT:-}" ] && [ "$1" = upgrade ] && [ -d /run/systemd/system ] ; then
	deb-systemd-invoke stop 'ssh.service' >/dev/null || true
fi
```

`/var/lib/dpkg/info/openssh-server.postinst`:

```sh
if deb-systemd-helper --quiet was-enabled ssh.socket; then
        deb-systemd-invoke restart ssh.socket
elif deb-systemd-helper --quiet was-enabled ssh.service; then
        deb-systemd-invoke restart ssh.service
fi
```

The stop always happens.  The restart takes the **first** branch that matches,
and on this host the first branch matches but does nothing:

1. `deb-systemd-helper was-enabled ssh.socket` reads the state file
   `/var/lib/systemd/deb-systemd-helper-enabled/ssh.socket.dsh-also`.  **That
   file does not exist here** (only `ssh.service.dsh-also` does).  In
   `/usr/bin/deb-systemd-helper`, `state_file_entries()` returns an empty list
   for a missing file, so `was_enabled()`'s loop never runs and it falls through
   to `return 1`, meaning "considering ssh.socket was-enabled".  The socket
   branch is taken.  VERIFIED by code read.
2. `deb-systemd-invoke restart ssh.socket` runs `systemctl is-enabled
   ssh.socket`, which returns `masked`.  That does not match `/enabled/` and the
   unit is not active, so `deb-systemd-invoke` prints "ssh.socket is a disabled
   or a static unit not running, not starting it." and restarts nothing.
   VERIFIED by code read (`/usr/bin/deb-systemd-invoke`, lines 110-145).
3. Because the `if` matched, the `elif` for `ssh.service` never runs.  sshd
   stays down.

This is consistent with the journal being completely silent about `ssh.socket`
for the entire boot (`journalctl -b -1 --grep "ssh\.socket"` returns nothing)
and with the absence of any `Starting ssh.service`.

The host is in that state because someone masked `ssh.socket` on 2026-04-04 to
get the classic always-listening sshd back.  That is a reasonable thing to do
and it worked fine, until the first `openssh-server` upgrade after it.  In the
retained apt history that upgrade is 2026-09-04, the only one since April.  The
bug fired the first time it could.

One piece of direct confirmation is still missing.  The `deb-systemd-invoke`
stderr line is captured in `/var/log/apt/term.log`, which is `root:adm 0640` and
was unreadable from the research account.  One command settles it:

```bash
sudo grep -n -B5 -A5 "ssh.socket" /var/log/apt/term.log
```

Everything else already points one way, but that line is the receipt.

## Why the machine went down at 19:05 on 2026-09-06 and came back at 19:07

Not a crash, not a power cut, not a kernel-upgrade reboot.  Someone went to the
machine physically, plugged in a USB keyboard and mouse at 19:00-19:01, and
pressed the SysRq sequence.  `/proc/sys/kernel/sysrq` is `176`
(16 sync + 32 remount-read-only + 128 reboot), so the R, E and I keys of REISUB
logged "This sysrq operation is disabled" while S produced `Emergency Sync` at
19:05:51, after which U and B took the machine down.  A correctly performed
manual hard reset.

The kernel upgrade did not cause the reboot.  `50unattended-upgrades` has no
`Automatic-Reboot` setting, so it defaults to false.  The 7.0.0-31 kernel was
installed at 06:18 on the 6th and simply sat there.  The kernel changed across
the reboot only because the reboot happened to come 13 hours later.

## A second, separate defect that turned two minutes into a day

The console GUI is in a restart loop and has been for at least the whole
retained journal.  gdm restarts the X greeter roughly every 15 seconds — 244,
244 and 246 restarts in the 12:00, 13:00 and 14:00 hours of 2026-09-05
(`journalctl -b -1 --grep "Running GNOME Shell \(using mutter"`).

Two consequences, both of which bit here:

- **The console was not usable either.**  That is why the recovery was a SysRq
  hard reset rather than sitting down and running `systemctl start ssh`.
- **It destroyed the forensic history.**  The journal is 3.1 GB for about nine
  days, roughly 440 MB a day of greeter-loop noise, rotating a 128 MB file every
  seven hours or so.  `journalctl --list-boots` claims boot -1 began 2026-08-28
  06:00, but `last -x reboot` puts it at **2026-08-13 13:12**; two weeks had
  already been rotated out.  Had the openssh upgrade been a week earlier, this
  note could not have been written.

## What was ruled out, and on what evidence

- **Resource exhaustion.**  `journalctl -b -1 --since "2026-09-05 16:00" --until
  "2026-09-06 19:10" -p err` returns eleven lines across 27 hours: nine snap
  firmware-notifier failures and two xfce4-notifyd failures.  No OOM kill, no
  fork or PID failure, no file-descriptor exhaustion.
- **Disk or filesystem.**  No I/O errors, no read-only remount, no SMART or NVMe
  messages anywhere in the window.
- **GPU or kernel fault.**  No Xid, no oops, no NVRM error.
- **The two ARD GPU jobs at 18:01 on 2026-09-05.**  They ran 18:01:53 to
  18:04:08 and both exited `rc=0`.  `sar` for the 18:00-18:10 bucket shows CPU
  2.15% user and 97.56% idle, memory 4.54% used with 242 GB available.  They
  also began **34 hours after** sshd had already stopped, so they cannot be a
  cause under any theory.
- **`ardx-watch.service`, the headless postrun, the campaign watcher, W&B
  sync.**  `ardx-watch.service` is a user unit and produced zero journal entries
  in the window.  Nothing ARD-related was running when sshd died on the 4th.
- **sshd being throttled, attacked or misconfigured.**  No
  `kex_exchange_identification`, MaxStartups, PAM or fork-failure messages.
  sshd's last words are the clean `Received signal 15; terminating`.
- **A second sshd on another port.**  Would not have helped: the same dpkg
  upgrade stops every instance from the same unit.  Explicitly rejected.

## Bounded fix

Each item needs administrator action; none of it is in this repository.

**1. Make the package upgrade restart sshd.  Required.**

`/etc/apt/apt.conf.d/99-ensure-sshd-running`:

```
DPkg::Post-Invoke {
  "if [ -d /run/systemd/system ] && systemctl is-enabled -q ssh.service && ! systemctl is-active -q ssh.service && ! systemctl is-active -q ssh.socket; then systemctl start ssh.service; fi || true";
};
```

It starts `ssh.service` only when that unit is already enabled and neither it
nor `ssh.socket` is currently serving.  It changes no sshd configuration, no
key, no port, no auth policy, and it never stops anything.

The alternative is to unmask `ssh.socket` and let Ubuntu's default socket
activation own port 22, which makes the postinst's first branch correct.  That
is cleaner but changes how sshd is started, with its own surprises (per-connection
unit instances, different `MaxStartups` semantics).  The APT hook is the smaller
change and is the recommendation.

Catches this failure and any recurrence.  Does not catch a kernel panic, power
loss, network failure, or the GUI loop.

**2. An off-host liveness alarm.  Required — this is what bounds the downtime.**

A systemd timer **on Ferret**, not on Hamster, that TCP-connects to Hamster port
22 every five minutes and notifies the phone after three consecutive failures.
It has to live on the other host, or it dies with the thing it watches.

Catches every failure mode in this class: sshd dead, kernel panic, power cut,
cable pulled.  Does not catch a reachable host whose campaign has silently
wedged.  Turns a 60-hour lockout into a 15-minute one.

**3. Bring up the BMC.  Recommended, needs lab approval.**

`/dev/ipmi0` exists and `ipmi_si`, `ipmi_ssif` and `ipmi_devintf` are loaded, so
this ASUS WS C621E SAGE board has a working baseboard controller.  If the
dedicated management port is cabled and given a lab IP, serial-over-LAN and iKVM
survive both a dead sshd and a broken GUI.  This is the only proposal here that
would have allowed a remote fix without walking to the machine.

**4. Do not blacklist `openssh-server` from unattended-upgrades.**  Skipping SSH
security updates on a machine reachable from the lab network trades a
recoverable outage for an unrecoverable one.  Fix the restart, keep the patches.

**5. Do not change the unattended kernel-upgrade policy because of this
incident.**  It is unrelated: the kernel upgrade landed 47 hours after the
lockout began and rebooted nothing.  Whether kernel upgrades should be scheduled
around long campaigns is a real question, but it needs its own evidence.

**6. Open a separate investigation for the gdm greeter loop, and cap the journal
meanwhile.**  Until the loop is fixed, `SystemMaxUse=1G` and a `RateLimitBurst`
in `/etc/systemd/journald.conf` would stop the noise from evicting the history
needed the next time something breaks.

## What was confirmed on 2026-09-06 evening

**The hook is installed on both hosts.** Not the version drafted above but a
broader one: it asks whether anything is listening on port 22 at all, rather
than whether `ssh.service` is enabled and inactive. The drafted condition would
have missed a postinst that *disabled* `ssh.service` as well as stopping it.
Source: `scripts/ardx/host-setup/ensure-sshd`.

**Ferret survived the identical upgrade for one reason, now verified.** It has
the state file Hamster lacks:

| | `/var/lib/systemd/deb-systemd-helper-enabled/ssh.socket.dsh-also` | `/etc/systemd/system/ssh.socket` | who holds port 22 |
| --- | --- | --- | --- |
| Hamster | absent | symlink to `/dev/null` (masked, dated 2026-04-04) | `ssh.service`, backlog 128 |
| Ferret | present, with `sockets.target.wants/ssh.socket` | not masked | `ssh.socket`, backlog 4096 |

On Ferret `was_enabled` finds recorded links and checks them, so the answer is
real. On Hamster it finds no state file, the loop never executes, and the
function returns true without having verified anything. That vacuous truth is
the whole bug.

**The migration branch was not involved.** Lines 148-160 of the postinst read
`if dpkg --compare-versions "$2" ge 1:9.3p1-1ubuntu3~ && ! systemctl --quiet
is-enabled ssh.socket; then :`. The upgrade was from `1:9.6p1-3ubuntu13.18`, so
the version test passes, and a masked socket is not enabled, so the negation
passes too. Both true means the `then` branch, which does nothing. It
deliberately leaves a masked socket alone. Only the restart branch at lines
221-224 misfires.

**Both BMCs are alive and on the network.** `ipmitool mc info` answers on both
hosts (ASUSTek, firmware 1.15, `Device Available: yes`), and the management LAN
is on **channel 8**, not channel 1:

| host | BMC address | MAC |
| --- | --- | --- |
| Hamster | 192.168.100.5 | 58:11:22:b4:44:8d |
| Ferret | 192.168.100.3 | 58:11:22:b4:44:39 |

Ports 623, 443 and 80 are open on each **as seen from the other host**. Neither
BMC answers from its own host, which is normal for a shared-NIC BMC and not a
fault. A console fallback therefore already exists for each machine; what
remains is credentials, which have to be set from the host with
`ipmitool user list 8`.

This closes item 3 of the bounded fix as "available", not "needs approval".

## Operational rule

Before starting a long unattended campaign on a host that will be left alone,
confirm three things in this order:

```bash
systemctl is-active ssh.service         # or ssh.socket, whichever this host uses
loginctl show-user "$USER" -p Linger    # must be Linger=yes, see note 0024
```

and confirm the off-host liveness alarm for that host has reported success
within the last interval.  `Linger=yes` already holds on Hamster.

## What this note does not establish

- The exact stderr text `deb-systemd-invoke` produced at 07:07:45 on 2026-09-04.
  It is in `/var/log/apt/term.log`, which needs `sudo` to read.  The mechanism is
  proven from the maintainer scripts and from the journal's silence; the log line
  would make it airtight.
- Why `ssh.socket` was masked on 2026-04-04, and by whom.  Only the symlink
  mtime survives.
- Why gdm is looping.  Not investigated beyond establishing the rate and the
  consequences.
- **Whether the same upgrade path will break Ferret.**  Ferret's `ssh.socket`
  mask state was not inspected.  Check it before the next `openssh-server`
  update reaches that host, or Ferret will be lost the same way:

  ```bash
  ssh Ferret 'systemctl is-enabled ssh.socket ssh.service; ls -l /etc/systemd/system/ssh.socket 2>/dev/null'
  ```

- Whether anything scientific was affected.  Nothing was: no run was in flight on
  2026-09-04 at 07:07, and the two jobs on the 5th completed `rc=0`.

## Corrections to what was believed before this investigation

1. The lockout began **2026-09-04 07:07:44**, about 35 hours earlier than the
   evening of 2026-09-05 when it was first noticed.
2. "The user-session systemd kept firing its timer through 2026-09-06 18:00:24,
   so boot -1 ended there" was an artifact of permissions.  Without the
   `systemd-journal` group, `journalctl` shows only user-journal entries, and the
   18:00:24 firmware notifier is simply the last such line.  With group access,
   boot -1 runs to **19:05:51** and ends in the SysRq sequence.
3. `journalctl --list-boots` reporting boot -1 from 2026-08-28 is the journal
   retention floor, not the boot.  `last -x reboot` puts it at 2026-08-13 13:12.
