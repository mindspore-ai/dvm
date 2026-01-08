#!/bin/bash
if [ $# -lt 1 ]; then
echo "Usage:"
echo "  1. setup environment: export DVM_SOC_NAME=Ascend<910B1|910B2|910B3|910B4>"
echo "  2. profiling: sh build.sh <case>"
exit 0
fi

TEST_FILE="${1%.cc}"
CFLAGS="--std=c++17 -ldl -Werror -Wall -I../../include -I${ASCEND_PATH}/include/experiment/runtime -I${ASCEND_PATH}/include/experiment/msprof/ -fvisibility=hidden"
LDFLGAS="-L${ASCEND_PATH}/lib64 -lascendcl -L${ASCEND_PATH}/tools/simulator/${DVM_SOC_NAME}/lib -lruntime_camodel"

echo "----------compile----------"

make -C ../../ dbg=1 -j32

echo "g++ ${CFLAGS} ${TEST_FILE}.cc ../../libdvm.a ${LDFLGAS} -o ${TEST_FILE}"

g++ ${CFLAGS} ${TEST_FILE}.cc ../../libdvm.a ${LDFLGAS} -o ${TEST_FILE}

echo "----------profiling----------"

msprof op simulator --application=./${TEST_FILE} --output=./profiling
