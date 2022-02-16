#!/bin/sh
set -euo pipefail

# download LAMA dataset
wget -nc https://dl.fbaipublicfiles.com/LAMA/data.zip
unzip data.zip
rm data.zip

pip install -r requirements.txt
