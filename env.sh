if [[ -z "${ASCEND_CUSTOM_PATH}" ]]; then
  if [[ -z "${ASCEND_PATH}" ]]; then
    if [ -d "/usr/local/Ascend/ascend-toolkit" ]; then
      export ASCEND_PATH="/usr/local/Ascend/ascend-toolkit"
    else
      export ASCEND_PATH="/usr/local/Ascend"
    fi
  fi
else
  export ASCEND_PATH="${ASCEND_CUSTOM_PATH}"
fi

echo "ASCEND_PATH: ${ASCEND_PATH}"
echo "ls ${ASCEND_PATH}"
ls ${ASCEND_PATH}

# auto config. DONOT config directly
export PY_INCLUDE=$(python -c "from sysconfig import get_paths as gp; print(gp()['include'])")
export PYTHONPATH=$(pwd)/python:${PYTHONPATH}
export PATH=${ASCEND_PATH}/latest/compiler/ccec_compiler/bin/:${ASCEND_PATH}/latest/compiler/bin:${PATH}
export LD_LIBRARY_PATH=${ASCEND_PATH}/latest/$(uname -m)-linux/lib64:${LD_LIBRARY_PATH}
