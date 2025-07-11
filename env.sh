#!/bin/bash

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
export PYTHONPATH=$(pwd)/python:${PYTHONPATH}

# Optional: Print environment summary
echo "Environment summary:"
echo "---------------------------------"
echo "ASCEND_PATH: ${ASCEND_PATH:-Not set}"
echo "PYBIND11_INCLUDES: ${PYBIND11_INCLUDES:-Not set}"
echo "---------------------------------"