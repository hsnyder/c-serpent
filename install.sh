#!/usr/bin/env bash
set -e
./build.sh
cp wrapgen ~/.local/bin
cp wrapgenx ~/.local/bin
cp wrapgen-fn ~/.local/bin
