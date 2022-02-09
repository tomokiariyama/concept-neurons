#!/bin/sh

export PYTHONIOENCODING=utf-8
export LC_ALL=ja_JP.UTF-8
export PYTHONHASHSEED=0

DATE=`date +%Y%m%d-%H%M`
echo $DATE

python make_graphs.py -dt ConceptNet -et subject \
       -rp "work/result/ConceptNet/subject/nt_4_at_0.2_mw_10" \
       --paper_figures

DATE=`date +%Y%m%d-%H%M`
echo $DATE
