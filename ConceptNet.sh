#!/bin/sh

source set_dataset_path.sh

export PYTHONIOENCODING=utf-8
export PYTHONHASHSEED=0

DATE=`date +%Y%m%d-%H%M`
echo $DATE

#python evaluate.py -dt ConceptNet -et object -nt 4 -at 0.2 --max_words 10 -ln conceptnet_obj -dp ${YOUR_DATA_DIRECTORY}
python evaluate.py -dt ConceptNet -et subject -nt 4 -at 0.2 --max_words 10 -ln conceptnet_sub -dp ${YOUR_DATA_DIRECTORY}

DATE=`date +%Y%m%d-%H%M`
echo $DATE
