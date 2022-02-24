#!/bin/bash

# assumes you have 'mlflow' installed in your conda environment
# also assumes that the mlrun dir is in the same folder as this script file

# find folder this script is located in - only breaks if symlinks are involved :/
# from here: https://stackoverflow.com/a/246128
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"
mlflow ui