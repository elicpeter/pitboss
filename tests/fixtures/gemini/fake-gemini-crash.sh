#!/bin/sh
# Exits non-zero with no JSON document at all and a stderr message — simulates
# the crash-before-startup case (e.g. binary couldn't open its config file).
# The parser must surface the exit code plus stderr tail.
set -eu

echo "Error: failed to read settings file" 1>&2
exit 1
