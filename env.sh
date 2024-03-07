if [[ -z "${ASCEND_CUSTOM_PATH}" ]]; then
  if [[ -z "${ASCEND_PATH}" ]]; then
    export ASCEND_PATH="/usr/local/Ascend"
  fi
else
  export ASCEND_PATH="${ASCEND_CUSTOM_PATH}"
fi

echo "ASCEND_PATH: ${ASCEND_PATH}"

# auto config. DONOT config directly
export PY_BIN_PATH=$(which python3.7m)
export PY_INCLUDE="${PY_BIN_PATH%bin/python3.7m}/include/python3.7m"
export PYTHONPATH=$(pwd)/python:${PYTHONPATH}
export PATH=${ASCEND_PATH}/latest/compiler/ccec_compiler/bin/:${ASCEND_PATH}/latest/compiler/bin:${PATH}
export LD_LIBRARY_PATH=${ASCEND_PATH}/latest/$(uname -m)-linux/lib64:${LD_LIBRARY_PATH}
