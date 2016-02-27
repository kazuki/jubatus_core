#!/bin/bash
set -e

THIRDPARTY_DIR="`pwd`/thirdparty"
mkdir -p "$THIRDPARTY_DIR"
ln -sf /usr/include/msgpack* "$THIRDPARTY_DIR/"

export CPLUS_INCLUDE_PATH="$THIRDPARTY_DIR"
emconfigure ./waf configure --regexp-library=none --prefix="`pwd`/installdir"
emmake ./waf clean
emmake ./waf build
emmake ./waf install
