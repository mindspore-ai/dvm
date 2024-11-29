#!/bin/bash
export DEVICE_ID=3
mpiexec -n 4 python -m pytest ./comm/test_allreduce.py
pytest
