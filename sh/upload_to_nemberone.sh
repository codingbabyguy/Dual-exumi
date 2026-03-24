#!/usr/bin/env bash
set -euo pipefail

# Upload a local folder to the LAN server.
# Server:
#   HostName: 10.108.16.211
#   Port: 2208
#   User: icrlab
# Default remote base dir:
#   /home/icrlab/tactile_work_Wy/data/
#
# Usage:
#   bash sh/upload_to_nemberone.sh <local_folder> [remote_subdir]
#
# Examples:
#   bash sh/upload_to_nemberone.sh data/session_20260323_101500
#   bash sh/upload_to_nemberone.sh data/my_dataset umi_exp

HOST="10.108.16.211"
PORT="2208"
USER="icrlab"
REMOTE_BASE="/home/icrlab/tactile_work_Wy/data"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash sh/upload_to_nemberone.sh <local_folder> [remote_subdir]"
  exit 1
fi

LOCAL_DIR="$1"
REMOTE_SUBDIR="${2:-}"

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "[ERROR] Local folder not found: $LOCAL_DIR"
  exit 1
fi

# Normalize path and remove trailing slash for predictable remote folder naming.
LOCAL_DIR="${LOCAL_DIR%/}"
LOCAL_BASENAME="$(basename "$LOCAL_DIR")"

if [[ -n "$REMOTE_SUBDIR" ]]; then
  REMOTE_TARGET="$REMOTE_BASE/${REMOTE_SUBDIR%/}"
else
  REMOTE_TARGET="$REMOTE_BASE"
fi

echo "[INFO] Local dir     : $LOCAL_DIR"
echo "[INFO] Remote target : $USER@$HOST:$REMOTE_TARGET"
echo "[INFO] Ensuring remote target exists..."
ssh -p "$PORT" "$USER@$HOST" "mkdir -p '$REMOTE_TARGET'"

if command -v rsync >/dev/null 2>&1; then
  echo "[INFO] Uploading with rsync (supports resume + progress)..."
  rsync -avh --progress \
    -e "ssh -p $PORT" \
    "$LOCAL_DIR/" \
    "$USER@$HOST:$REMOTE_TARGET/$LOCAL_BASENAME/"
else
  echo "[WARN] rsync not found, fallback to scp -r..."
  scp -P "$PORT" -r "$LOCAL_DIR" "$USER@$HOST:$REMOTE_TARGET/"
fi

echo "[DONE] Upload completed."
