#!/bin/bash
if [ $# -lt 2 ]; then
echo "Usage:"
echo "  1. setup environment: export ASCEND_CUSTOM_PATH=<path_to_ascend_toolkit>"
echo "  2. profiling: sh build.sh <case> <910B1|910B2|910B3|910B4>"
exit 0
fi

export DVM_SOC_NAME=Ascend${2}
TEST_FILE="${1%.cc}"
CFLAGS="--std=c++17 -ldl -Werror -Wall -I../../include -I${ASCEND_PATH}/include/experiment/runtime -I${ASCEND_PATH}/include/experiment/msprof/ -fvisibility=hidden"
LDFLGAS="-L${ASCEND_PATH}/tools/simulator/${DVM_SOC_NAME}/lib -lruntime_camodel"

export LD_LIBRARY_PATH=${ASCEND_PATH}/tools/simulator/${DVM_SOC_NAME}/lib:${LD_LIBRARY_PATH}

echo "----------compile----------"

make -C ../../ libdvm.a sim=1 dbg=1 -j8

echo "g++ ${CFLAGS} ${TEST_FILE}.cc ../../libdvm.a ${LDFLGAS} -o ${TEST_FILE} -ldl"

g++ ${CFLAGS} ${TEST_FILE}.cc ../../libdvm.a ${LDFLGAS} -o ${TEST_FILE}

echo "----------profiling----------"

msprof op simulator --application=./${TEST_FILE} --output=./profiling
