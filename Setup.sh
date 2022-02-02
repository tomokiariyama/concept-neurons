#!/bin/sh

source set_dataset_path.sh

# LAMAデータセットのwget文
mkdir -p ${YOUR_DATA_DIRECTORY}
wget -nc https://dl.fbaipublicfiles.com/LAMA/data.zip -P ${YOUR_DATA_DIRECTORY}
unzip ${YOUR_DATA_DIRECTORY}/data.zip -d ${YOUR_DATA_DIRECTORY}
rm ${YOUR_DATA_DIRECTORY}/data.zip

# (使用する仮想環境に移動した上で)仮想環境に必要なモジュールをインストール
pip install -r requirements.txt
