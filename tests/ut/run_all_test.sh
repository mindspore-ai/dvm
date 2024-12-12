#!/bin/bash
export DEVICE_ID=3
pytest
export DEVICE_IDS=0,1,2,3
python -m pytest ./comm/test_allreduce.py
