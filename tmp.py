# -*- coding: utf-8 -*-

from knowledge_neurons import KnowledgeNeurons, initialize_model_and_tokenizer, model_type, ALL_MODELS
from data import extract_raw_dataset_from_jsonlines, extract_matched_dataset
import random
import logzero
from logzero import logger
import argparse
import os
import pathlib
import numpy as np
import torch
import datetime
from collections import defaultdict
import nltk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 全ての知識ニューロンの場所を可視化する
root_save_path = os.path.join("/", "home", "ariyama", "sftp_pycharm_sync", "concept_neurons", "work", "figure", "ConceptNet", "subject")
#root_save_path = os.path.join("work", "figures", "ConceptNet", "subject")
layer_of_others_pos_neurons_list = [7, 11, 10, 0, 5, 8, 9]
#layer_of_total_neurons_dict = defaultdict(int)
#for refined_neurons_list in place_of_total_neurons_dict.values():
#    for place_of_neuron in refined_neurons_list:
#        layer_of_total_neurons_list.append(place_of_neuron[0])
#        layer_of_total_neurons_dict[place_of_neuron[0]] += 1

#print(f'layer_of_total_neurons_dict: {layer_of_total_neurons_dict}\n')  # キー：層数、バリュー：その層にある知識ニューロンの数
ndarray_of_layer_of_others_pos_neurons = np.array(layer_of_others_pos_neurons_list)
plt.title('place_of_layer_of_others_pos_neurons')
plt.xlabel('place_of_layer')
plt.ylabel('freq')
plt.hist(ndarray_of_layer_of_others_pos_neurons, range=(0, 11), bins=12)
save_path = os.path.join(root_save_path, f"place_of_layer_of_others_pos_neurons.png")
plt.savefig(save_path)
print(f"figure is saved in {save_path}")
plt.close()