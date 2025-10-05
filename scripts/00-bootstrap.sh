#!/usr/bin/env bash
# Compatibility wrapper for old script name
exec bash "$(dirname "$0")/setup/bootstrap.sh" "$@"