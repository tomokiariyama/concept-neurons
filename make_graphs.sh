#!/bin/sh

export PYTHONIOENCODING=utf-8
export LC_ALL=ja_JP.UTF-8
export PYTHONHASHSEED=0

DATE=`date +%Y%m%d-%H%M`
echo $DATE

python make_graphs.py -dt ConceptNet -et subject -gt "pos"
python make_graphs.py -dt ConceptNet -et subject -gt "kitten"

DATE=`date +%Y%m%d-%H%M`
echo $DATE
