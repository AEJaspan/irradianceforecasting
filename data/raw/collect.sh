#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo $SCRIPT_DIR
if ! command -v zenodo_get &> /dev/null
then
    echo "zenodo_get could not be found. Resorting to wget."
    wget -i ${SCRIPT_DIR}/urls.txt -P $SCRIPT_DIR
else
    echo "Downloading source data using zenodo_get"
    zenodo_get 2826939 -o $SCRIPT_DIR
    if [ ! -f ${SCRIPT_DIR}/urls.txt ]; then
        echo "Collecting list of URL paths for zenodo downloads."
        zenodo_get 2826939 -w ${SCRIPT_DIR}/urls.txt
    fi
fi