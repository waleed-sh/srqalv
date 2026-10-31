#!/bin/sh
set -eu

exec "${PYTHON:-python}" -m srqalv experiment=all "$@"
