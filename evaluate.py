# -*- coding: utf-8 -*-

from knowledge_neurons import KnowledgeNeurons, initialize_model_and_tokenizer, model_type, ALL_MODELS
from data import extract_dicts_from_jsonlines, format_data_with_entity_type
import random
import logzero
from logzero import logger
import argparse
import os
import pathlib
import numpy as np
import torch
import datetime


def make_parser():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--seed', help='乱数のシード値, default=42', type=int, default=42)
    parser.add_argument('-mn', '--model_name', help='使用するニューラルモデルの名前, default="bert-base-uncased"', type=str, default="bert-base-uncased")
    parser.add_argument('-dt', '--dataset_type', help="使用するデータセットを指定します", default="ConceptNet")
    parser.add_argument('-et', '--entity_type', help="解析する対象を指定します", default="subject")
    parser.add_argument('-nt', '--number_of_templates', help='この数以上のテンプレートを持つエンティティのみを解析対象とします, default=4', type=int, default=4)
    parser.add_argument('--local_rank', help="local rank for multigpu processing, default=0", type=int, default=0)
    parser.add_argument('-ln', '--logfile_name', help="ログを保存するファイル名を指定します, default='run'", type=str, default="run")
    parser.add_argument('-bs', '--batch_size', help="", type=int, default=20)
    parser.add_argument('--steps', help="number of steps in the integrated grad calculation", type=int, default=20)
    parser.add_argument('-at', '--adaptive_threshold', help="the threshold value", type=float, default=0.3)
    parser.add_argument('-sp', '--sharing_percentage', help="the threshold for the sharing percentage", type=float, default=0.5)
    parser.add_argument('--rawdataset_filename', type=str, default='test.jsonl')
    parser.add_argument('--max_words', type=int, default=15)
    parser.add_argument('-dp', '--dataset_path', help="LAMAデータセットがダウンロードされているディレクトリのパス", required=True)

    return parser.parse_args()


def make_log(args):
    dt_now = datetime.datetime.now()
    str_dt_now = "_" + str(dt_now.year) + "-" + str(dt_now.month) + "-" + str(dt_now.day) + "-" + \
                 str(dt_now.hour) + "-" + str(dt_now.minute) + "-" + str(dt_now.second)
    log_file_name = args.logfile_name + str_dt_now + ".log"
    os.makedirs(os.path.join("log", args.dataset_type, args.entity_type), exist_ok=True)
    log_file_path = os.path.join("log", args.dataset_type, args.entity_type, log_file_name)
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
    logger.info('the number of minimum templates is ' + str(args.number_of_templates))


def main():
    args = make_parser()

    make_log(args)

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

    # make dataset with parsed settings
    dataset_path = os.path.join(args.dataset_path, "data", args.dataset_type, args.rawdataset_filename)
    raw_dataset = extract_dicts_from_jsonlines(dataset_path)
    matched_dataset = format_data_with_entity_type(args.entity_type, raw_dataset, args.number_of_templates, args.max_words)

    logger.info('Number of entities covered this time: ' + str(len(matched_dataset.keys())))
    logger.info('')

    # 今回の条件でデータセットとして用いたエンティティをファイルに書きだす
    os.makedirs(os.path.join("work", "entities"), exist_ok=True)
    save_entities_path = os.path.join("work", "entities", f"{args.dataset_type}_{args.entity_type}.txt")
    if matched_dataset:
        with open(save_entities_path, mode="w") as fi:
            fi.write(f"# All entities appeared in {args.dataset_type}_{args.entity_type}\n")
            fi.write("\n")
            for entity in matched_dataset.keys():
                fi.write(entity + "\n")

    # to find the knowledge neurons, we need the same 'facts' expressed in multiple different ways, and a ground truth
    result_path = os.path.join("work", "result", args.dataset_type, args.entity_type)
    os.makedirs(result_path, exist_ok=True)
    suppress_relevant = os.path.join(result_path, "suppress_activation_and_relevant_prompts.txt")
    suppress_unrelated = os.path.join(result_path, "suppress_activation_and_unrelated_prompts.txt")
    enhance_relevant = os.path.join(result_path, "enhance_activation_and_relevant_prompts.txt")
    enhance_unrelated = os.path.join(result_path, "enhance_activation_and_unrelated_prompts.txt")

    with open(suppress_relevant, mode="w") as sr_fi, open(suppress_unrelated, mode="w") as su_fi, \
         open(enhance_relevant, mode="w") as er_fi, open(enhance_unrelated, mode="w") as enu_fi:
        for file_pointer in [sr_fi, su_fi, er_fi, enu_fi]:
            file_pointer.write("<LEGEND> GROUND_TRUTH: {number_of_refined_neurons}: {before_gt_prob}, {before_argmax_completion}, {before_argmax_prob}: {after_gt_prob}, {after_argmax_completion}, {after_argmax_prob}\n")
            file_pointer.write("\n")

        total_templates = 0
        for entity, templates in matched_dataset.items():
            TEXTS = list(templates)
            TEXT = TEXTS[0]
            GROUND_TRUTH = entity

            try:
                assert len(TEXTS) >= args.number_of_templates, "テンプレートの最小数を満たしていないエンティティが含まれてしまっています"
            except AssertionError as err:
                logger.info('AssertionError: ' + str(err))
                logger.info(
                    'The number of minimum templates: ' + str(args.number_of_templates) + ', the length of this ground_truth templates: ' + str(len(TEXTS)))
                logger.info('Ground_truth: ' + GROUND_TRUTH)
                logger.info('The number of related templates: ' + str(len(TEXTS)))
                logger.info(f'Templates: {TEXTS}')
                logger.info('')

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

            # logger.info(torch.cuda.memory_summary())
            unrelated_prompt = "[MASK] is the official language of the solomon islands"
            unrelated_ground_truth = "english"
            target = "london"

            # suppress the activations at the refined neurons + test the effect on a relevant prompt
            # 'results_dict' is a dictionary containing the probability of the ground truth being generated before + after modification, as well as other info
            # 'unpatch_fn' is a function you can use to undo the activation suppression in the model.
            # By default, the suppression is removed at the end of any function that applies a patch, but you can set 'undo_modification=False',
            # run your own experiments with the activations / weights still modified, then run 'unpatch_fn' to undo the modifications
            logger.info('suppress the activations at the refined neurons + test the effect on a relevant prompt')
            results_dict, unpatch_fn = kn.suppress_knowledge(
                TEXT, GROUND_TRUTH, refined_neurons
            )

            # logger.info(f'results_dict: {results_dict}')
            # logger.info(f'unpatch_fn: {unpatch_fn}')
            sr_fi.write(f'{GROUND_TRUTH}:: {number_of_refined_neurons}:: {results_dict["before"]["gt_prob"]}, {results_dict["before"]["argmax_completion"]}, {results_dict["before"]["argmax_prob"]}:: {results_dict["after"]["gt_prob"]}, {results_dict["after"]["argmax_completion"]}, {results_dict["after"]["argmax_prob"]}:: {TEXT}\n')


            # suppress the activations at the refined neurons + test the effect on an unrelated prompt
            logger.info('suppress the activations at the refined neurons + test the effect on an unrelated prompt')
            results_dict, unpatch_fn = kn.suppress_knowledge(
                unrelated_prompt,
                unrelated_ground_truth,
                refined_neurons,
            )
            su_fi.write(f'{GROUND_TRUTH}:: {number_of_refined_neurons}:: {results_dict["before"]["gt_prob"]}, {results_dict["before"]["argmax_completion"]}, {results_dict["before"]["argmax_prob"]}:: {results_dict["after"]["gt_prob"]}, {results_dict["after"]["argmax_completion"]}, {results_dict["after"]["argmax_prob"]}:: {TEXT}\n')


            # enhance the activations at the refined neurons + test the effect on a relevant prompt
            logger.info('enhance the activations at the refined neurons + test the effect on a relevant prompt')
            results_dict, unpatch_fn = kn.enhance_knowledge(TEXT, GROUND_TRUTH, refined_neurons)
            er_fi.write(f'{GROUND_TRUTH}:: {number_of_refined_neurons}:: {results_dict["before"]["gt_prob"]}, {results_dict["before"]["argmax_completion"]}, {results_dict["before"]["argmax_prob"]}:: {results_dict["after"]["gt_prob"]}, {results_dict["after"]["argmax_completion"]}, {results_dict["after"]["argmax_prob"]}:: {TEXT}\n')


            # enhance the activations at the refined neurons + test the effect on an unrelated prompt
            logger.info('enhance the activations at the refined neurons + test the effect on an unrelated prompt')
            results_dict, unpatch_fn = kn.enhance_knowledge(
                unrelated_prompt,
                unrelated_ground_truth,
                refined_neurons,
            )
            enu_fi.write(f'{GROUND_TRUTH}:: {number_of_refined_neurons}:: {results_dict["before"]["gt_prob"]}, {results_dict["before"]["argmax_completion"]}, {results_dict["before"]["argmax_prob"]}:: {results_dict["after"]["gt_prob"]}, {results_dict["after"]["argmax_completion"]}, {results_dict["after"]["argmax_prob"]}:: {TEXT}\n')

            logger.debug('')

    logger.info(f"average templates per GROUND_TRUTH = {total_templates / len(matched_dataset)}")

    logger.debug('script done!')
    logger.debug('')


if __name__=='__main__':
    main()
