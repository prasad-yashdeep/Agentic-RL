#!/usr/bin/env bash
# Sync run artifacts to the capacity tier (node05 /local — wipeable on
# reprovision; anything irreplaceable also lives on NFS). Run FROM node08.
set -uo pipefail
DEST="node05:/local/yashp/tarl"
ssh -o ConnectTimeout=10 node05 'mkdir -p /local/yashp/tarl/jobs' \
    || { echo "node05 unreachable"; exit 1; }
rsync -a --exclude "*.lock" "$HOME/terminal-agent-rl/jobs/" "$DEST/jobs/"
rsync -a "$HOME/terminal-agent-rl/experiments/" "$DEST/experiments/"
echo "synced $(date "+%H:%M:%S")"
