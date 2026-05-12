#!/bin/bash

CUR_SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# ----------------------------
# Arg parse (only supports --key=value)
#   --ascend_path=/path/to/ascend-toolkit
#   --simulator_name=910B1
# ----------------------------
ASCEND_PATH_ARG=""
SIMULATOR_NAME=""

while [ $# -gt 0 ]; do
  case "$1" in
    --ascend_path=*)
      ASCEND_PATH_ARG="${1#*=}"
      shift
      ;;
    --simulator_name=*)
      SIMULATOR_NAME="${1#*=}"
      shift
      ;;
    -h|--help)
      echo "Usage:"
      echo "  source env.sh --ascend_path=/path/to/ascend-toolkit"
      echo "  source env.sh --ascend_path=/path/to/ascend-toolkit --simulator_name=910B1"
      return 0 2>/dev/null || exit 0
      ;;
    *)
      echo "WARNING: Unknown argument ignored: $1"
      shift
      ;;
  esac
done


# ----------------------------
# Set Pybind11 includes if not already set
# ----------------------------
if [[ -z "${PYBIND11_INCLUDES}" ]]; then
  export PYBIND11_INCLUDES=$(python -m pybind11 --includes 2>/dev/null)
  if [ $? -ne 0 ]; then
    echo "WARNING: Failed to get pybind11 includes. Please ensure pybind11 is installed."
  fi
fi

# ----------------------------
# Set Ascend environment variables
# Priority:
# 1) --ascend_path
# 2) ASCEND_TOOLKIT_HOME / ASCEND_CUSTOM_PATH
# 3) /usr/local/Ascend/ascend-toolkit
# ----------------------------
if [[ -z "${ASCEND_TOOLKIT_HOME}" ]]; then
  if [[ -z "${ASCEND_CUSTOM_PATH}" ]]; then
    if [ -d "/usr/local/Ascend/ascend-toolkit" ]; then
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
    fi
  else
    source "${ASCEND_CUSTOM_PATH}/ascend-toolkit/set_env.sh"
  fi
fi

# ----------------------------
# Set additional environment variables
# ----------------------------
if [ -n "${ASCEND_TOOLKIT_HOME}" ]; then
  export ASCEND_PATH="${ASCEND_TOOLKIT_HOME}"
elif [ -n "${ASCEND_PATH_ARG}" ]; then
  export ASCEND_PATH="${ASCEND_PATH_ARG}"
fi

export PYTHONPATH="${CUR_SCRIPT_PATH}:${CUR_SCRIPT_PATH}/python:${PYTHONPATH}"

# ----------------------------
# Simulator mode (explicit flag)
# ----------------------------
SIMULATOR_NAMES=("910B1" "910B2" "910B3" "910B4")

if [ -n "${SIMULATOR_NAME}" ]; then
  echo "Enabling simulator mode..."
  export DVM_SOC_SIMU="Ascend${SIMULATOR_NAME}"
  export DEVICE_ID=0
  echo "DVM_SOC_SIMU: ${DVM_SOC_SIMU}"
  echo "Note: For ESL Model mode:"
  echo "      export LD_LIBRARY_PATH=/path/to/your/esl_lib:\$LD_LIBRARY_PATH"
  echo "Note: For regular simulation mode:"
  echo "      export LD_LIBRARY_PATH=\${ASCEND_PATH}/tools/simulator/\${DVM_SOC_SIMU}/lib:\$LD_LIBRARY_PATH"
fi

# ----------------------------
# Optional: Print environment summary
# ----------------------------
echo "Environment summary:"
echo "---------------------------------"
echo "ASCEND_PATH: ${ASCEND_PATH:-Not set}"
echo "SIMULATOR_NAME: ${SIMULATOR_NAME:-Disabled}"
echo "PYBIND11_INCLUDES: ${PYBIND11_INCLUDES:-Not set}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "---------------------------------"
