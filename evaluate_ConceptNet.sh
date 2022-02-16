#!/bin/sh

source dataset_path.sh

export PYTHONIOENCODING=utf-8
export PYTHONHASHSEED=0

DATE=`date +%Y%m%d-%H%M`
echo $DATE

python evaluate.py -dt ConceptNet -et subject -nt 4 -at 0.2 --mw 10 -ln conceptnet_sub -dp ${DATA_DIRECTORY}

DATE=`date +%Y%m%d-%H%M`
echo $DATE
