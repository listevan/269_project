#!/usr/bin/bash
# Keeps the experiment 3 run alive: cron runs this every 5 minutes. If the run has finished (exit file) it does
# nothing; if the driver is running it does nothing; otherwise it stops any orphaned stage process (a killed driver
# leaves vLLM's EngineCore holding GPU memory) and relaunches the driver, which resumes: finished stages are skipped
# and training continues from its last checkpoint (neg_train.py --ckpt-every). Launched from cron, the driver lives
# outside this terminal's process group, which a Claude Code session restart kills (2026-11-24, twice).
# Remove the crontab line when the run is done:  crontab -l | grep -v exp3_watchdog | crontab -
ROOT=/home/steven-li/269_project
TAG=2026-11-24
cd "$ROOT" || exit 1
[ -f "runs/${TAG}_rules.exit" ] && exit 0
if pgrep -f "run_rules.py --tag ${TAG}" > /dev/null; then exit 0; fi
# orphans of the previous driver only: at this point no driver is running
for pat in "src/neg_train.py --run" "src/rules_eval.py --run" "src/rules_probe.py --run" "src/teacher_audit.py --run"; do
    pkill -f "$pat" 2>/dev/null
done
for p in $(ps -eo pid,comm | awk '$2 ~ /^VLLM/ {print $1}'); do kill "$p" 2>/dev/null; done
sleep 3
echo "===== $(date '+%F %T') watchdog relaunch" >> "runs/${TAG}_rules.out"
export HOME=/home/steven-li PATH=/home/steven-li/.local/bin:/usr/local/bin:/usr/bin:/bin
setsid -f bash -c "/home/steven-li/.local/bin/uv run run_rules.py --tag ${TAG} >> runs/${TAG}_rules.out 2>&1; echo \$? > runs/${TAG}_rules.exit"
