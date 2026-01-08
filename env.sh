#!/bin/bash

CUR_SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# Set Ascend environment variables
if [[ -z "${ASCEND_TOOLKIT_HOME}" ]]; then
    if [[ -z "${ASCEND_CUSTOM_PATH}" ]]; then
        if [ -d "/usr/local/Ascend/ascend-toolkit" ]; then
            source /usr/local/Ascend/ascend-toolkit/set_env.sh
        fi
    else
        source ${ASCEND_CUSTOM_PATH}/ascend-toolkit/set_env.sh
    fi
fi

# Set Pybind11 includes if not already set
if [[ -z "${PYBIND11_INCLUDES}" ]]; then
    # Safely get pybind11 includes
    export PYBIND11_INCLUDES=$(python -m pybind11 --includes 2>/dev/null)
    if [ $? -ne 0 ]; then
        echo "WARNING: Failed to get pybind11 includes. Please ensure pybind11 is installed."
    fi
fi

# Set additional environment variables
export ASCEND_PATH=${ASCEND_TOOLKIT_HOME}
export PYTHONPATH=${CUR_SCRIPT_PATH}:${CUR_SCRIPT_PATH}/python:${PYTHONPATH}

# Set simulator mode
SOC_NAMES=("910B1" "910B2" "910B3" "910B4")
if [ $# -eq 1 ]; then
  soc_name="$1"
  is_valid=0
  for value in "${SOC_NAMES[@]}"; do
    if [ "$1" = "$value" ]; then
      is_valid=1
      break
    fi
  done

  if [ $is_valid -eq 1 ]; then
    echo "Enabling simulator mode..."
    export DVM_SOC_NAME=Ascend${soc_name}
    echo "DVM_SOC_NAME: ${DVM_SOC_NAME}"
    export DEVICE_ID=0
    echo "Note: For ESL Model mode, please configure the following:"
    echo "      export LD_LIBRARY_PATH=/path/to/your/esl_lib:\$LD_LIBRARY_PATH"
    echo "Note: For regular simulation mode, please configure the following:"
    echo "      export LD_LIBRARY_PATH=\${ASCEND_PATH}/tools/simulator/\${DVM_SOC_NAME}/lib:\$LD_LIBRARY_PATH"
  else
    echo "Invalid param: ${soc_name}"
    echo "Valid param：${SOC_NAMES[*]}"
  fi
fi

CCEC_BUILD_DATE=$(ccec --version | head -n 1)
if [[ ${CCEC_BUILD_DATE:0:10} > "2025-12-00" ]]; then
  export CANN_VER_85=1
  echo "CANN version is 8.5+"
fi

# Optional: Print environment summary
echo "Environment summary:"
echo "---------------------------------"
echo "ASCEND_PATH: ${ASCEND_PATH:-Not set}"
echo "PYBIND11_INCLUDES: ${PYBIND11_INCLUDES:-Not set}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "---------------------------------"