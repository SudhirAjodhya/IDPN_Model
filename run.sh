#!/bin/bash

PYTHON="./torch_env/Scripts/python.exe" # points to local env

case "$1" in
    test)
        $PYTHON run_test.py
        ;;
    train)
        $PYTHON train.py
        ;;
    *)
        echo "Usage: ./run.sh {test|train}"
        exit 1
esac