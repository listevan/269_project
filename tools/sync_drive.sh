#!/usr/bin/env bash
# Copy the code to the Drive 269_project folder, where the Colab notebook runs it. No GitHub token needed.
#
#   tools/sync_drive.sh            after any edit the VM should see
#
# Writes <Drive>/269_project/code/src (plus a VERSION stamp the notebook prints) and a copy of the notebook at
# <Drive>/269_project/token_replay_pilot.ipynb to open with Colab. Drive for desktop uploads both. The Drive
# folder is found under ~/Library/CloudStorage; set DRIVE_269 to override.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
if [ -z "${DRIVE_269:-}" ]; then
    DRIVE_269="$(find "$HOME/Library/CloudStorage"/GoogleDrive-*/"My Drive" -maxdepth 5 -type d -name 269_project \
                 -path '*CS 269*' 2>/dev/null | head -1)"
fi
[ -n "$DRIVE_269" ] && [ -d "$DRIVE_269" ] || { echo "Drive 269_project folder not found; set DRIVE_269=<path>" >&2; exit 1; }

mkdir -p "$DRIVE_269/code"
rsync -a --delete --exclude '__pycache__' "$HERE/src" "$DRIVE_269/code/"
cp "$HERE/notebooks/token_replay_pilot.ipynb" "$DRIVE_269/token_replay_pilot.ipynb"
stamp="$(git -C "$HERE" log -1 --format='%h %s' 2>/dev/null || echo 'no git')"
if [ -n "$(git -C "$HERE" status --porcelain -- src 2>/dev/null)" ]; then stamp="$stamp +uncommitted src edits"; fi
echo "$stamp | synced $(date '+%Y-%m-%d %H:%M')" > "$DRIVE_269/code/VERSION"
echo "synced -> $DRIVE_269"
echo "         code/VERSION: $(cat "$DRIVE_269/code/VERSION")"
