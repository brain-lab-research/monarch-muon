#!/usr/bin/env bash
# bash run.sh ARM DATA_DIR [extra src/main.py flags...]   (see train.py / README.md)
exec python3 "$(dirname "$0")/train.py" "$@"
