#!/usr/bin/env bash
# Run on YOUR MAC. Creates the project folder on the remote machine and uploads everything needed.
#   bash scripts/upload_to_remote.sh                       # defaults below
#   REMOTE=user@host REMOTE_DIR=amazon_ml_2026 bash scripts/upload_to_remote.sh
# You will be asked for the remote password (once for mkdir, once for rsync).
# Tip: run `ssh-copy-id user@host` once to avoid password prompts.
set -euo pipefail
REMOTE="${REMOTE:?set REMOTE=user@host}"
REMOTE_DIR="${REMOTE_DIR:-amazon_ml_2026}"      # relative to the remote home directory
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for d in dataset/train dataset/test utils code/business_entity_resolution scripts; do
  [[ -e "$d" ]] || { echo "missing $d — run this from the challenge folder"; exit 1; }
done

echo "== creating ~/$REMOTE_DIR on $REMOTE"
ssh "$REMOTE" "mkdir -p ~/$REMOTE_DIR"

SRC=(dataset utils code scripts docs reports Documentation_template.md README.md)
for d in offline_wheels hf_cache; do
  if [[ -d "$d" ]]; then SRC+=("$d"); fi
done
echo "== uploading: ${SRC[*]} (unchanged files are skipped)"
rsync -avzP \
  --exclude '.venv*' --exclude 'artifacts' --exclude 'output*' --exclude '.locks' \
  --exclude 'logs' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' \
  --exclude '*_submission.zip' \
  "${SRC[@]}" "$REMOTE:$REMOTE_DIR/"

echo
echo "Upload complete. Now on the remote machine:"
echo "  ssh $REMOTE"
echo "  cd ~/$REMOTE_DIR && bash scripts/start_remote.sh"
