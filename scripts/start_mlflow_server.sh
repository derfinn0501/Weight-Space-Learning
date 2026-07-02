#!/usr/bin/env bash
set -euo pipefail

exec python scripts/start_mlflow_server.py "$@"
