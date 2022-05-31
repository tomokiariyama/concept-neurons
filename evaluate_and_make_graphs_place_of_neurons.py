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


def make_parser():
    parser = argparse.ArgumentParser(description='Conduct experiments')

    parser.add_argument('--seed', help='the seed of random numbers. default=42', type=int, default=42)
    parser.add_argument('-mn', '--model_name', help='the name of neural model. default="bert-base-uncased"', type=str,
                        default="bert-base-uncased")
    parser.add_argument('-dt', '--dataset_type', help="designate the dataset", default="ConceptNet",
                        choices=["ConceptNet", "TREx"])
    parser.add_argument('-et', '--entity_type', help="designate entity type which is masked", default="subject",
                        choices=["subject", "object"])
    parser.add_argument('-nt', '--number_of_templates',
                        help='the minimum number of templates which each entity have. default=4', type=int, default=4)
    parser.add_argument('--local_rank', help="local rank for multigpu processing, default=0", type=int, default=0)
    parser.add_argument('-ln', '--logfile_name', help="designate the file name of log. default='run'", type=str,
                        default="run")
    parser.add_argument('-bs', '--batch_size', help="", type=int, default=20)
    parser.add_argument('--steps', help="number of steps in the integrated grad calculation", type=int, default=20)
    parser.add_argument('-at', '--adaptive_threshold', help="the threshold value", type=float, default=0.3)
    parser.add_argument('-sp', '--sharing_percentage', help="the threshold for the sharing percentage", type=float,
                        default=0.5)
    parser.add_argument('--rawdataset_filename', type=str, default='test.jsonl')
    parser.add_argument('-mw', '--max_words', help="the maximum number of words which each template can have", type=int,
                        default=15)
    parser.add_argument('-dp', '--dataset_path', help="the path for the LAMA dataset", default=".")

    return parser.parse_args()


def main():
    args = make_parser()

    dt_now = datetime.datetime.now()
    str_dt_now = "_" + str(dt_now.year) + "-" + str(dt_now.month) + "-" + str(dt_now.day) + "-" + \
                 str(dt_now.hour) + "-" + str(dt_now.minute) + "-" + str(dt_now.second)
    log_directory = os.path.join("log", args.dataset_type, args.entity_type)
    os.makedirs(log_directory, exist_ok=True)
    log_file_name = args.logfile_name + str_dt_now + ".log"
    log_file_path = os.path.join(log_directory, log_file_name)
    if not os.path.isfile(log_file_path):
        log_file = pathlib.Path(log_file_path)
        log_file.touch()
    logger = logzero.setup_logger(
        logfile=log_file_path,
        disableStderrLogger=False
    )
    print('log is saved in ' + log_file_path)
    logger.info('--------start of script--------')

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    logger.info('random seed is ' + str(args.seed))
    logger.info('model name is ' + args.model_name)
    logger.info('dataset type is ' + args.dataset_type)
    logger.info('entity type is ' + args.entity_type)
    logger.info('the number of templates each entity at least have is ' + str(args.number_of_templates))

    torch.cuda.set_device(args.local_rank)

    # first initialize some hyperparameters
    MODEL_NAME = args.model_name

    # these are some hyperparameters for the integrated gradients step
    BATCH_SIZE = args.batch_size
    STEPS = args.steps  # number of steps in the integrated grad calculation
    ADAPTIVE_THRESHOLD = args.adaptive_threshold  # in the paper, they find the threshold value `t` by multiplying the max attribution score by some float - this is that float.
    P = args.sharing_percentage  # the threshold for the sharing percentage

    # setup model & tokenizer
    model, tokenizer = initialize_model_and_tokenizer(MODEL_NAME)

    # initialize the knowledge neuron wrapper with your model, tokenizer and a string expressing the type of your model ('gpt2' / 'gpt_neo' / 'bert')
    kn = KnowledgeNeurons(model, tokenizer, model_type=model_type(MODEL_NAME))

    # make dataset by the conditions
    dataset_path = os.path.join(args.dataset_path, "data", args.dataset_type, args.rawdataset_filename)
    raw_dataset = extract_raw_dataset_from_jsonlines(dataset_path)
    matched_dataset = extract_matched_dataset(raw_dataset, args.entity_type, args.number_of_templates, args.max_words)

    logger.info('Number of entities covered this time: ' + str(len(matched_dataset.keys())))
    logger.info('')

    # Write out the entities used as the dataset in this condition to a file.
    os.makedirs(os.path.join("work", "entities"), exist_ok=True)
    save_entities_path = os.path.join("work", "entities",
                                      f"{args.dataset_type}_{args.entity_type}_nt_{args.number_of_templates}_at_{args.adaptive_threshold}_mw_{args.max_words}.txt")
    if matched_dataset:
        with open(save_entities_path, mode="w") as fi:
            fi.write(f"# All entities appeared in {args.dataset_type}_{args.entity_type}\n")
            fi.write("\n")
            for entity in matched_dataset.keys():
                fi.write(entity + "\n")

    # to find the knowledge neurons, we need the same 'facts' expressed in multiple different ways, and a ground truth
    result_path = os.path.join("work", "result", args.dataset_type, args.entity_type,
                               f"nt_{args.number_of_templates}_at_{args.adaptive_threshold}_mw_{args.max_words}")
    os.makedirs(result_path, exist_ok=True)
    place_of_neurons = os.path.join(result_path, "place_of_neurons.txt")

    ng_words = ("marijuana", "grenade", "guns", "bigotry", "rifle", "revolver", "pistol", "destruction",
                "terrorism", "fart", "farting", "urinate", "urinating", "ejaculate", "orgasm", "penis", "copulate",
                "copulating", "flirt", "flirting", "sex", "reproduce", "reproducing", "fuck", "pee", "poop", "shit")
    with open(place_of_neurons, mode="w") as fi:
        fi.write("<LEGEND> GROUND_TRUTH:: {number_of_refined_neurons}:: {lists of layer num and neuron's place}")
        fi.write("\n")

        total_templates = 0
        place_of_total_neurons_dict = defaultdict(list)
        place_of_noun_neurons_dict = defaultdict(list)
        place_of_verb_adjective_adverb_neurons_dict = defaultdict(list)
        place_of_others_pos_neurons_dict = defaultdict(list)
        for entity, templates in matched_dataset.items():
            TEXTS = list(templates)
            TEXT = TEXTS[0]
            GROUND_TRUTH = entity

            logger.info("Ground Truth: " + GROUND_TRUTH)
            logger.info('The number of related templates: ' + str(len(TEXTS)))
            logger.info(f'Templates: {TEXTS}')
            logger.info("")

            total_templates += len(TEXTS)

            # use the integrated gradients technique to find some refined neurons for your set of prompts
            refined_neurons = kn.get_refined_neurons(
                TEXTS,
                GROUND_TRUTH,
                p=P,
                batch_size=BATCH_SIZE,
                steps=STEPS,
                coarse_adaptive_threshold=ADAPTIVE_THRESHOLD,
            )

            logger.info('refining done')
            number_of_refined_neurons = len(refined_neurons)

            if GROUND_TRUTH not in ng_words:
                # 後で確認する用の、精製された知識ニューロンの場所の記録
                fi.write(f'{GROUND_TRUTH}:: {number_of_refined_neurons}::  {refined_neurons}\n')

                # 品詞全てをひっくるめた知識ニューロンの場所の辞書
                place_of_total_neurons_dict[GROUND_TRUTH] = refined_neurons

                # 品詞ごとに分類してから知識ニューロンの場所をそれぞれの品詞用の辞書に格納
                no_mask_sentence = TEXT.replace('[MASK]', GROUND_TRUTH)
                for morph in nltk.pos_tag(nltk.word_tokenize(no_mask_sentence)):
                    if morph[0] == GROUND_TRUTH:
                        if morph[1] in ["NN", "NNS", "NNP", "NNPS"]:  # 現在の概念が名詞の場合
                            place_of_noun_neurons_dict[GROUND_TRUTH] = refined_neurons
                            break
                        elif morph[1] in ["JJ", "JJR", "JJS", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP",
                                          "VBZ"]:  # 現在の概念が形容詞・副詞・動詞の場合
                            place_of_verb_adjective_adverb_neurons_dict[GROUND_TRUTH] = refined_neurons
                            break
                        else:  # その他の品詞の場合
                            place_of_others_pos_neurons_dict[GROUND_TRUTH] = refined_neurons
                            break

            logger.debug('')

    logger.info(f"place_of_total_neurons_dict: {place_of_total_neurons_dict}\n")
    logger.info(f"place_of_noun_neurons_dict: {place_of_noun_neurons_dict}\n")
    logger.info(f"place_of_verb_adjective_adverb_neurons_dict: {place_of_verb_adjective_adverb_neurons_dict}\n")
    logger.info(f"place_of_others_pos_neurons_dict: {place_of_others_pos_neurons_dict}\n")

    # 全ての知識ニューロンの場所を可視化する
    root_save_path = os.path.join("/", "home", "ariyama", "sftp_pycharm_sync", "concept_neurons", "work", "figure",
                                  "ConceptNet", "subject")

    layer_of_total_neurons_list = []
    layer_of_total_neurons_dict = defaultdict(int)
    for refined_neurons_list in place_of_total_neurons_dict.values():
        for place_of_neuron in refined_neurons_list:
            layer_of_total_neurons_list.append(place_of_neuron[0])
            layer_of_total_neurons_dict[place_of_neuron[0]] += 1

    print(f'layer_of_total_neurons_dict: {layer_of_total_neurons_dict}\n')  # キー：層数-1、バリュー：その層にある知識ニューロンの数
    ndarray_of_layer_of_total_neurons = np.array(layer_of_total_neurons_list)
    plt.title('place_of_layer_of_total_neurons')
    plt.xlabel('place_of_layer')
    plt.ylabel('freq')
    plt.hist(ndarray_of_layer_of_total_neurons, range=(0, 11), bins=12)
    save_path = os.path.join(root_save_path, f"place_of_layer_of_total_neurons.png")
    plt.savefig(save_path)
    print(f"figure is saved in {save_path}")
    plt.close()

    # 名詞の知識ニューロンの場所を可視化する
    layer_of_noun_neurons_list = []
    layer_of_noun_neurons_dict = defaultdict(int)
    for refined_neurons_list in place_of_noun_neurons_dict.values():
        for place_of_neuron in refined_neurons_list:
            layer_of_noun_neurons_list.append(place_of_neuron[0])
            layer_of_noun_neurons_dict[place_of_neuron[0]] += 1

    print(f'layer_of_noun_neurons_dict: {layer_of_noun_neurons_dict}\n')  # キー：層数、バリュー：その層にある知識ニューロンの数
    ndarray_of_layer_of_noun_neurons = np.array(layer_of_noun_neurons_list)
    plt.title('place_of_layer_of_noun_neurons')
    plt.xlabel('place_of_layer')
    plt.ylabel('freq')
    plt.hist(ndarray_of_layer_of_noun_neurons, range=(0, 11), bins=12)
    save_path = os.path.join(root_save_path, f"place_of_layer_of_noun_neurons.png")
    plt.savefig(save_path)
    print(f"figure is saved in {save_path}")
    plt.close()

    # 動詞・形容詞・副詞の知識ニューロンの場所を可視化する
    layer_of_verb_adjective_adverb_neurons_list = []
    layer_of_verb_adjective_adverb_neurons_dict = defaultdict(int)
    for refined_neurons_list in place_of_verb_adjective_adverb_neurons_dict.values():
        for place_of_neuron in refined_neurons_list:
            layer_of_verb_adjective_adverb_neurons_list.append(place_of_neuron[0])
            layer_of_verb_adjective_adverb_neurons_dict[place_of_neuron[0]] += 1

    print(
        f'layer_of_verb_adjective_adverb_neurons_dict: {layer_of_verb_adjective_adverb_neurons_dict}\n')  # キー：層数、バリュー：その層にある知識ニューロンの数
    ndarray_of_layer_of_verb_adjective_adverb_neurons = np.array(layer_of_verb_adjective_adverb_neurons_list)
    plt.title('place_of_layer_of_verb_adjective_adverb_neurons')
    plt.xlabel('place_of_layer')
    plt.ylabel('freq')
    plt.hist(ndarray_of_layer_of_verb_adjective_adverb_neurons, range=(0, 11), bins=12)
    save_path = os.path.join(root_save_path, f"place_of_layer_of_verb_adjective_adverb_neurons.png")
    plt.savefig(save_path)
    print(f"figure is saved in {save_path}")
    plt.close()

    # その他の品詞のの知識ニューロンの場所を可視化する
    layer_of_others_pos_neurons_list = []
    layer_of_others_pos_neurons_dict = defaultdict(int)
    for refined_neurons_list in place_of_others_pos_neurons_dict.values():
        for place_of_neuron in refined_neurons_list:
            layer_of_others_pos_neurons_list.append(place_of_neuron[0])
            layer_of_others_pos_neurons_dict[place_of_neuron[0]] += 1

    print(
        f'layer_of_others_pos_neurons_dict: {layer_of_others_pos_neurons_dict}\n')  # キー：層数、バリュー：その層にある知識ニューロンの数
    ndarray_of_layer_of_others_pos_neurons = np.array(layer_of_others_pos_neurons_list)
    plt.title('place_of_layer_of_others_pos_neurons')
    plt.xlabel('place_of_layer')
    plt.ylabel('freq')
    plt.hist(ndarray_of_layer_of_others_pos_neurons, range=(0, 11), bins=12)
    save_path = os.path.join(root_save_path, f"place_of_layer_of_others_pos_neurons.png")
    plt.savefig(save_path)
    print(f"figure is saved in {save_path}")
    plt.close()

    # 一つの概念あたりの平均テンプレート数を表示
    logger.info(f"average templates per GROUND_TRUTH = {total_templates / len(matched_dataset)}")

    logger.debug('script done!')
    logger.debug('')


if __name__ == '__main__':
    main()
