#!/bin/sh

export PYTHONIOENCODING=utf-8
export PYTHONHASHSEED=0

DATE=`date +%Y%m%d-%H%M`
echo $DATE

python evaluate_place_of_neurons.py -dt ConceptNet -et subject -nt 4 -at 0.2 -mw 10 -ln place_of_neurons -dp "/work02/ariyama/nlp2022_repo_testdir"

DATE=`date +%Y%m%d-%H%M`
echo $DATE