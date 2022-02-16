#!/bin/sh
Set -euo pipefail

source dataset_path.sh

# download LAMA dataset
mkdir -p ${DATA_DIRECTORY}
wget -nc https://dl.fbaipublicfiles.com/LAMA/data.zip -P ${DATA_DIRECTORY}
unzip ${DATA_DIRECTORY}/data.zip -d ${DATA_DIRECTORY}
rm ${DATA_DIRECTORY}/data.zip

pip install -r requirements.txt