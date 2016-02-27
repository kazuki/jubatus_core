#!/bin/bash
#em++ -std=c++11 -o hoge.js -s EXPORT_ALL=1 -s DEMANGLE_SUPPORT=1 -s EXCEPTION_DEBUG=1 -I./thirdparty -I./installdir/include wrapper.cpp build/libjubatus_core.a build/jubatus/util/*/*.a build/jubatus/util/*.a
em++ -std=c++11 -o hoge.js -O3 --llvm-lto 1 --memory-init-file 0 -I./thirdparty -I./installdir/include wrapper.cpp build/libjubatus_core.a build/jubatus/util/*/*.a build/jubatus/util/*.a
