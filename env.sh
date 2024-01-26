export TEST_TARGET=910

# auto config. DONOT config directly
export PY_BIN_PATH=$(which python3.7m)
export PY_INCLUDE="${PY_BIN_PATH%bin/python3.7m}/include/python3.7m"
export PYTHONPATH=$(pwd)/python:${PYTHONPATH}
export PATH=${ASCEND_CUSTOM_PATH}/latest/compiler/ccec_compiler/bin/:${ASCEND_CUSTOM_PATH}/latest/compiler/bin:${PATH}
export LD_LIBRARY_PATH=${ASCEND_CUSTOM_PATH}/latest/$(uname -m)-linux/lib64:${LD_LIBRARY_PATH}
#export LD_LIBRARY_PATH=${ASCEND_CUSTOM_PATH}/latest/toolkit/tools/simulator/Ascend910B1/lib:${LD_LIBRARY_PATH}
