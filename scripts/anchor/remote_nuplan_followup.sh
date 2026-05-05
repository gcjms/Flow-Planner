#!/usr/bin/env bash
set -euo pipefail

NUROOT="${NUROOT:-/root/autodl-tmp/nuplan_official}"
MINI_URL="${MINI_URL:-https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_mini.zip}"
VAL_URL="${VAL_URL:-https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_val.zip}"
SMOKE_SCRIPT="${SMOKE_SCRIPT:-/root/autodl-tmp/Flow-Planner-anchor-runtime/scripts_anchor_run_closed_loop_smoke_anchor.sh}"
LOG_DIR="${LOG_DIR:-$NUROOT}"

mkdir -p "$NUROOT" "$LOG_DIR"
cd "$NUROOT"

while pgrep -f "$MINI_URL" >/dev/null 2>&1; do
  sleep 30
done

if [[ ! -f nuplan-v1.1_mini.zip ]]; then
  echo "mini zip missing at $NUROOT/nuplan-v1.1_mini.zip"
  exit 2
fi

if [[ ! -d "$NUROOT/nuplan-v1.1" && ! -d "$NUROOT/public/nuplan-v1.1" ]]; then
  unzip -q nuplan-v1.1_mini.zip -d "$NUROOT/"
fi

rm -f nuplan-v1.1_mini.zip

if ! pgrep -f "$VAL_URL" >/dev/null 2>&1; then
  screen -dmS nuplan_val_dl bash -lc "cd '$NUROOT' && wget -c '$VAL_URL' -O nuplan-v1.1_val.zip >> val_download.log 2>&1"
fi

if [[ -x "$SMOKE_SCRIPT" ]]; then
  "$SMOKE_SCRIPT" > "$LOG_DIR/closed_loop_smoke.log" 2>&1
fi
