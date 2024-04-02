#!/bin/bash
if [ $# -lt 2 ]; then
echo "Usage:"
echo "  1. setup environment: export ASCEND_CUSTOM_PATH=<path_to_ascend_toolkit>"
echo "  2. profiling: sh build.sh <case> <910B1|910B2|910B3|910B4>"
exit 0
fi

export ASCEND_TOOLKIT_HOME=$ASCEND_CUSTOM_PATH/latest
export ASCEND_PATH=${ASCEND_CUSTOM_PATH}
export DVM_SOC_NAME=Ascend${2}
TEST_FILE="${1%.cc}"
CFLAGS="--std=c++17 -Werror -Wall -I../../include -fvisibility=hidden"
LDFLGAS="-L${ASCEND_TOOLKIT_HOME}/tools/simulator/${DVM_SOC_NAME}/lib -lruntime_camodel"

export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/$(uname -m)-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/tools/aml/lib64:${ASCEND_TOOLKIT_HOME}/tools/aml/lib64/plugin:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/tools/simulator/${DVM_SOC_NAME}/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
export PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:${ASCEND_TOOLKIT_HOME}/tools/ccec_compiler/bin:$PATH
export TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit
export ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}

echo "----------compile----------"

make -C ../../ libdvm.a sim=1 dbg=1 -j8

g++ ${CFLAGS} ${TEST_FILE}.cc ../../libdvm.a ${LDFLGAS} -o ${TEST_FILE}

echo "----------profiling----------"

msprof op simulator --application=./${TEST_FILE} --output=./profiling
