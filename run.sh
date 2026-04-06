#!/bin/bash

PYTHON="./torch_env/Scripts/python.exe" # points to local env

case "$1" in
    test)
        $PYTHON run_test.py
        ;;
    train)
        $PYTHON train.py
        ;;
    downsampler_test)
        $PYTHON -m src.models.downsamplers.implementation_test
        ;;
    *)
        echo "Usage: ./run.sh {test|train|max_pool_implementation_test}"
        exit 1
esac