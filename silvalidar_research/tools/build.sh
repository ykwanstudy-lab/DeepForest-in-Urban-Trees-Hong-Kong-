#!/bin/bash
# Build innoextract 1.13.0 from source with the local GCC 12 + boost 1.74 deps.
set -e
cd /workspace/silvalidar_research/tools

SRC=innoextract-1.13.0
INC="-I$SRC/src -isystem $PWD/deps/root/usr/include"
DEFS="-D_GNU_SOURCE=1 -DBOOST_SYSTEM_DISABLE_THREADS -DBOOST_ALL_NO_LIB"
CXXFLAGS="-std=c++17 -O1 -fvisibility=hidden -fvisibility-inlines-hidden $DEFS $INC"

rm -rf obj
mkdir -p obj

find "$SRC/src" -name '*.cpp' ! -name test.cpp ! -path '*/util/windows.cpp' > srcs.txt
echo "$SRC/src/release.cpp" >> srcs.txt

xargs -P 3 -I FILE sh -c "g++ $CXXFLAGS -c FILE -o obj/\$(echo FILE | tr / _).o" < srcs.txt

g++ obj/*.o -L libs -o innoextract \
    -lboost_iostreams -lboost_filesystem -lboost_program_options -lboost_date_time \
    -llzma -lz

echo "BUILD OK"
./innoextract --version
