#!/bin/bash

BASE_HASH=$(tail -n 1 prebuild/lib_info.txt | awk '{print $3}')
CHANGE_CNT=$(git diff --name-only ${BASE_HASH} src/isa.h src/vm_aic.cce src/vm_aiv.cce | wc -l)
if [[ $CHANGE_CNT > "0" ]]; then
    echo "================================"
    echo "NEW CHANGE: g_vkernel_c220_bin"
    echo "================================"
    git log ${BASE_HASH}..HEAD src/isa.h src/vm_aic.cce src/vm_aiv.cce | cat
fi
CHANGE_CNT=$(git diff --name-only ${BASE_HASH} src/isa.h src/vm_aic_c310.cce src/vm_aiv_c310.cce | wc -l)
if [[ $CHANGE_CNT > "0" ]]; then
    echo "================================"
    echo "NEW CHANGE: g_vkernel_c310_bin"
    echo "================================"
    git log ${BASE_HASH}..HEAD src/isa.h src/vm_aic_c310.cce src/vm_aiv_c310.cce | cat
fi