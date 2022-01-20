#!/bin/sh

#export YOUR_DATA_DIRECTORY="please set your data path"
source set_dataset_path.sh

# LAMAデータセットのwget文
mkdir -p ${YOUR_DATA_DIRECTORY}
wget -nc https://dl.fbaipublicfiles.com/LAMA/data.zip -P ${YOUR_DATA_DIRECTORY}
unzip ${YOUR_DATA_DIRECTORY}/data.zip -d ${YOUR_DATA_DIRECTORY}
rm ${YOUR_DATA_DIRECTORY}/data.zip

<< COMMENT
ln -s ${YOUR_DATA_DIRECTORY} dataset_dir
mkdir -p dataset_dir
wget -O dataset_dir/. https://dl.fbaipublicfiles.com/LAMA/data.zip
unzip ${YOUR_DATA_DIRECTORY}/data.zip
rm ${YOUR_DATA_DIRECTORY}/data.zip
COMMENT

# (使用する仮想環境に移動した上で)仮想環境に必要なモジュールをインストール
pip install -r requirements.txt
