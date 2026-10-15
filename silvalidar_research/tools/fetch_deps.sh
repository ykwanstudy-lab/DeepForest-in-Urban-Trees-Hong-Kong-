#!/bin/bash
# Fetch and unpack the build/run dependencies for tools/build.sh.
# Everything lands in this directory (tools/) and is git-ignored.
set -e
cd "$(dirname "$0")"

BASE=https://deb.debian.org/debian/pool/main
mkdir -p debs deps/root root

urls=(
  "$BASE/b/boost1.74/libboost1.74-dev_1.74.0+ds1-21_amd64.deb"
  "$BASE/b/boost1.74/libboost-filesystem1.74.0_1.74.0+ds1-21_amd64.deb"
  "$BASE/b/boost1.74/libboost-program-options1.74.0_1.74.0+ds1-21_amd64.deb"
  "$BASE/b/boost1.74/libboost-iostreams1.74.0_1.74.0+ds1-21_amd64.deb"
  "$BASE/b/boost1.74/libboost-date-time1.74.0_1.74.0+ds1-21_amd64.deb"
  "$BASE/z/zlib/zlib1g-dev_1.2.13.dfsg-1_amd64.deb"
  "$BASE/x/xz-utils/liblzma-dev_5.8.1-1+deb13u1_amd64.deb"
)

for u in "${urls[@]}"; do
  f="debs/$(basename "$u")"
  if [ ! -s "$f" ]; then
    echo "fetching $(basename "$u")"
    curl -sSL --max-time 120 -o "$f" "$u"
  fi
done

# Boost runtime shared libraries (needed to run the built binary)
for d in debs/libboost-*1.74.0_*.deb; do dpkg -x "$d" root/; done

# Boost / zlib / liblzma headers
for d in debs/libboost1.74-dev_*.deb debs/zlib1g-dev_*.deb debs/liblzma-dev_*.deb; do
  dpkg -x "$d" deps/root/
done

# innoextract source (>= 1.11 is required for Inno Setup 6.7 installers)
if [ ! -d innoextract-1.13.0 ]; then
  curl -sSL --max-time 120 -o src.tar.gz \
    https://codeload.github.com/crazy-max/innoextract/tar.gz/refs/tags/v1.13.0
  tar xzf src.tar.gz
fi

# Hand-written stand-ins for the files CMake would normally generate
cp patches/configure.hpp patches/release.cpp innoextract-1.13.0/src/

echo "dependencies ready"
