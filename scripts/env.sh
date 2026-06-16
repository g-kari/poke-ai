#!/usr/bin/env bash
# Source this to get a Python environment where numpy and kaggle_environments
# actually import. The nix-built python interpreter under devbox needs
# libstdc++ and libz on LD_LIBRARY_PATH to load numpy's compiled extensions.
#
# Usage:
#   source scripts/env.sh
#   python3 selfplay_test.py 4
#
# Or as a wrapper:
#   scripts/run.sh python3 -m train.reinforce --episodes 500

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "$ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "$ROOT/.venv/bin/activate"
fi

_find_lib() {
    # Print one absolute path containing $1 from the nix store, newest first.
    find /nix/store -maxdepth 3 -name "$1" 2>/dev/null | head -1
}

_libstdcpp="$(_find_lib libstdc++.so.6)"
_libz="$(_find_lib libz.so.1)"

if [ -n "$_libstdcpp" ]; then
    export LD_LIBRARY_PATH="$(dirname "$_libstdcpp"):${LD_LIBRARY_PATH:-}"
fi
if [ -n "$_libz" ]; then
    export LD_LIBRARY_PATH="$(dirname "$_libz"):${LD_LIBRARY_PATH:-}"
fi

# WSL2: expose libcuda.so.1 + other GPU shims provided by the Windows driver
if [ -d /usr/lib/wsl/lib ]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
fi

unset _libstdcpp _libz _find_lib
