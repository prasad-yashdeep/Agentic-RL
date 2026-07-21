#!/usr/bin/env bash
# M0.3 — user-space apptainer install (idempotent). Run ON node08.
set -euo pipefail
DEST="${1:-$HOME/apptainer}"
if [ -x "$DEST/bin/apptainer" ]; then
    echo "apptainer already installed: $("$DEST/bin/apptainer" --version)"
    exit 0
fi
curl -fsSL https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh \
    -o /tmp/install-unprivileged.sh
bash /tmp/install-unprivileged.sh "$DEST"
"$DEST/bin/apptainer" --version
echo "OK — add $DEST/bin to PATH (scripts/env.sh does this)"
