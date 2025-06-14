#!/bin/bash
set -e

APP_DIR=/opt/DataSetKurator
SERVICE_FILE=/etc/systemd/system/datasetkurator.service

check_deps() {
  for cmd in git python3 ffmpeg systemctl; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      echo "Missing dependency: $cmd" >&2
      exit 1
    fi
  done
}

create_venv() {
  python3 -m venv "$APP_DIR/venv"
  "$APP_DIR/venv/bin/pip" install --upgrade pip
  "$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt"
}

# Verify that required ML models can be loaded before enabling the service.
check_models() {
  "$APP_DIR/venv/bin/python" <<'PY'
import sys
import open_clip

repo = "SmilingWolf/wd-v1-4-swinv2-tagger-v3"
try:
    open_clip.create_model_and_transforms(f"hf-hub:{repo}")
except Exception as exc:
    sys.stderr.write(f"Failed to download tagger model: {exc}\n")
    sys.exit(1)
PY
}

install_app() {
  check_deps
  mkdir -p "$APP_DIR"
  # copy the entire repository including the .git folder so updates work via git
  rsync -a --delete "$(dirname "$0")/" "$APP_DIR/"
  create_venv
  check_models

  cat > "$SERVICE_FILE" <<SERVICE
[Unit]
Description=DataSetKurator service
After=network.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/venv/bin/python $APP_DIR/app.py
Restart=always
User=$(whoami)

[Install]
WantedBy=multi-user.target
SERVICE

  systemctl daemon-reload
  systemctl enable datasetkurator.service
  systemctl start datasetkurator.service
  echo "Installed and started DataSetKurator"
}

deinstall_app() {
  systemctl stop datasetkurator.service 2>/dev/null || true
  systemctl disable datasetkurator.service 2>/dev/null || true
  rm -f "$SERVICE_FILE"
  rm -rf "$APP_DIR"
  systemctl daemon-reload
  echo "DataSetKurator removed"
}

update_app() {
  if [ ! -d "$APP_DIR" ]; then
    echo "Application not installed in $APP_DIR" >&2
    exit 1
  fi
  check_deps
  if [ -d "$APP_DIR/.git" ]; then
    git -C "$APP_DIR" pull
  else
    # if the installation lacks a git repository, sync the current script
    rsync -a --delete "$(dirname "$0")/" "$APP_DIR/"
  fi
  create_venv
  check_models
  systemctl restart datasetkurator.service
  echo "DataSetKurator updated"
}

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root" >&2
  exit 1
fi

case "$1" in
  --install)
    install_app
    ;;
  --Deinstall|--deinstall)
    deinstall_app
    ;;
  --Update|--update)
    update_app
    ;;
  *)
    echo "Usage: $0 [--install|--Deinstall|--Update]" >&2
    exit 1
    ;;
esac
