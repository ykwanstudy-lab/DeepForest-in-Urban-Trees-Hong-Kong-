#!/bin/bash
# Build innoextract 1.13.0 from source with the local GCC 12 + boost 1.74 deps.
# Run fetch_deps.sh first (it also applies patches/configure.hpp and patches/release.cpp,
# which replace the files CMake would normally generate).
set -e
cd /workspace/silvalidar_research/tools

SRC=innoextract-1.13.0
INC="-I$SRC/src -isystem $PWD/deps/root/usr/include"
DEFS="-D_GNU_SOURCE=1 -DBOOST_SYSTEM_DISABLE_THREADS -DBOOST_ALL_NO_LIB"
CXXFLAGS="-std=c++17 -O1 -fvisibility=hidden -fvisibility-inlines-hidden $DEFS $INC"

# Linker names for the boost/zlib/lzma shared objects extracted into root/
mkdir -p libs
ln -sf ../root/usr/lib/x86_64-linux-gnu/libboost_iostreams.so.1.74.0 libs/libboost_iostreams.so
ln -sf ../root/usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.74.0 libs/libboost_filesystem.so
ln -sf ../root/usr/lib/x86_64-linux-gnu/libboost_program_options.so.1.74.0 libs/libboost_program_options.so
ln -sf ../root/usr/lib/x86_64-linux-gnu/libboost_date_time.so.1.74.0 libs/libboost_date_time.so
ln -sf /lib/x86_64-linux-gnu/libz.so.1 libs/libz.so
ln -sf /lib/x86_64-linux-gnu/liblzma.so.5 libs/liblzma.so

rm -rf obj
mkdir -p obj

# util/test.cpp needs BUILD_TESTS, util/windows.cpp is Win32-only
find "$SRC/src" -name '*.cpp' ! -name test.cpp ! -path '*/util/windows.cpp' > srcs.txt
echo "$SRC/src/release.cpp" >> srcs.txt

xargs -P 3 -I FILE sh -c "g++ $CXXFLAGS -c FILE -o obj/\$(echo FILE | tr / _).o" < srcs.txt

g++ obj/*.o -L libs -o innoextract \
    -lboost_iostreams -lboost_filesystem -lboost_program_options -lboost_date_time \
    -llzma -lz

echo "BUILD OK"
LD_LIBRARY_PATH=root/usr/lib/x86_64-linux-gnu ./innoextract --version
