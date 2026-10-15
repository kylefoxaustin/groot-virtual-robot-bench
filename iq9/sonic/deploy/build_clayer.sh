#!/bin/bash
# The MuJoCo simulator talks DDS through the cyclonedds Python package (0.10.2, what unitree_sdk2py uses). There is no aarch64
# wheel for it, so build its C extension (_clayer) on the x86 host against the libddsc that upstream ships for aarch64 in
# gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty (the same DDS the deploy links), and the board Python's headers.
# Output: $IQ9_SONIC_WORK/cyclonedds-0.10.2/cyclonedds/ (pure-Python package + _clayer); stage_board.sh ships it with libddsc.
# Prereq: `git lfs pull` in $SONIC_REPO (the aarch64 .a/.so files are LFS objects; without it they are text pointer files).
set -e
U=$SONIC_DEPLOY/thirdparty/unitree_sdk2/thirdparty
PYTAG=$(echo $IQ9_PYVER | tr -d .)
INC=$IQ9_SONIC_WORK/py${PYTAG}inc
mkdir -p $INC && rsync -a $IQ9_HOST:$IQ9_PY_ROOT/include/python$IQ9_PYVER/ $INC/
cd $IQ9_SONIC_WORK
[ -d cyclonedds-0.10.2 ] || (curl -sL -o cyclonedds-0.10.2.tar.gz https://files.pythonhosted.org/packages/source/c/cyclonedds/cyclonedds-0.10.2.tar.gz && tar xzf cyclonedds-0.10.2.tar.gz)
cd cyclonedds-0.10.2
file -L $U/lib/aarch64/libddsc.so.0 | grep -q ELF || { echo "libddsc is not an ELF file: run git lfs pull in \$SONIC_REPO"; exit 1; }
$IQ9_CC -shared -fPIC -O2 -I$INC -I$U/include -Iclayer clayer/cdrkeyvm.c clayer/pysertype.c clayer/typeser.c \
  -L$U/lib/aarch64 -lddsc -Wl,-rpath,'$ORIGIN/../cyclonedds.libs' -o cyclonedds/_clayer.cpython-${PYTAG}-aarch64-linux-gnu.so
ls -la cyclonedds/_clayer*.so
