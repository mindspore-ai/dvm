#!/bin/bash
export DEVICE_ID=3
pytest
python -m pytest ./comm/test_allreduce.py
