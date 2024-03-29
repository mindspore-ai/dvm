#!/bin/bash
echo "ASCEND_TOOLKIT_HOME: ${ASCEND_TOOLKIT_HOME}"
export DVM_SOC_NAME=Ascend${2}
TEST_FILE="${1%.cc}"
CFLAGS="--std=c++17 -Werror -Wall -I../../include -fvisibility=hidden"
LDFLGAS="-L${ASCEND_PATH}/latest/toolkit/tools/simulator/${DVM_SOC_NAME}/lib -lruntime_camodel"

if [ -n "$2" ]; then
  export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH}
  export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/tools/aml/lib64:${ASCEND_TOOLKIT_HOME}/tools/aml/lib64/plugin:${LD_LIBRARY_PATH}
  export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/tools/simulator/${DVM_SOC_NAME}/lib:${LD_LIBRARY_PATH}
  export PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
  export PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:${ASCEND_TOOLKIT_HOME}/tools/ccec_compiler/bin:$PATH
  export TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit
  export ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}
fi

echo "----------compile----------"

make -C ../../ libdvm.a sim=1 dbg=1 -j8

g++ ${CFLAGS} ${TEST_FILE}.cc ../../libdvm.a ${LDFLGAS} -o ${TEST_FILE}

echo "----------profiling----------"

msprof op simulator --application=./${TEST_FILE} --output=./profiling
