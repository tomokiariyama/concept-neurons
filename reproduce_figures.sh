#!/bin/sh

source set_dataset_path.sh

./setup.sh
./evaluate_ConceptNet.sh
./make_graphs.sh
