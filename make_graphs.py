import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import traceback
import sys
import nltk
import math
from matplotlib import rcParams
import japanize_matplotlib


def make_parser():
    parser = argparse.ArgumentParser(description="Make results into a graph.")

    parser.add_argument("-dt", "--dataset_type", help="dataset type", type=str, default="ConceptNet")
    parser.add_argument("-et", "--entity_type", help="entity type", type=str, default="subject")
    #parser.add_argument("-gt", "--graph_type", help="Designate making graphs including 'kitten' entity, or analyzing the effect of each POS.", type=str, default="kitten", choices=["kitten", "pos", "other"])
    parser.add_argument("--for_any_entity",
                        help="(optional) Shifts the entities displayed in the graph arbitrarily, set to an int type between 0 and 729. Notice that this flag can be set if --graph_type is 'other'.",
                        type=int, default=1
                        )
    parser.add_argument("--article_figures", help="only generate the figures in the article", action='store_true')

    return parser.parse_args()


def sns_settings():
    sns.set(
        context="paper",
        style="whitegrid",
        palette=sns.color_palette("Set1", 24),
        font_scale=4,
        rc={"lines.linewidth": 6, 'grid.linestyle': '--'},
        font='IPAexGothic'  # Japanese font
        )


def sns_settings2():
    sns.set(
        context="paper",
        style="whitegrid",
        palette=sns.dark_palette("palegreen", n_colors=2, reverse=True),
        font_scale=4,
        rc={"lines.linewidth": 6, 'grid.linestyle': '--'},
        font='IPAexGothic'
        )


def extract_dict_from_result_file(result_file_path):
    with open(result_file_path) as re_fi:
        prob_change_dict = defaultdict(tuple)
        ng_word_list = ["marijuana", "grenade", "guns", "bigotry", "rifle", "revolver", "pistol", "destruction", "terrorism", "fart", "farting", "urinate", "urinating", "ejaculate", "orgasm", "penis", "copulate", "copulating", "flirt", "flirting", "sex", "reproduce", "reproducing", "fuck", "pee", "poop", "shit"]
        for line in re_fi:
            if not line.startswith(("<", "\n")):
                ground_truth, number_of_refined_neurons, before, after, prompt = line.split(":: ")
                if ground_truth not in ng_word_list:
                    before_data = before.split(", ")
                    after_data = after.split(", ")
                    prob_change_dict[ground_truth] = (number_of_refined_neurons, before_data[0], after_data[0], prompt)

    return prob_change_dict


def extract_gt_type_dict(prob_change_dict, ground_truth_type):
    if ground_truth_type == "entity":
        # 名詞と判定されたground truthのみグラフ化の対象とする
        matched_pos_list = ["NN", "NNS", "NNP", "NNPS"]
    elif ground_truth_type == "concept":
        # 形容詞・副詞・動詞と判定されたground truthのみグラフ化の対象とする
        matched_pos_list = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    elif ground_truth_type == "others":
        matched_pos_list = ["CC", "CD", "DT", "EX", "FW", "IN", "LS", "MD", "PDT", "POS", "PRP", "PRP$", "RP", "SYM", "TO", "UH", "WDT", "WP", "WP$", "WRB", ",", "."]
    else:
        assert False, "need to debug, around 'ground_truth_type'"

    particular_gt_type_dict = defaultdict(tuple)
    for ground_truth, info_tuple in prob_change_dict.items():
        no_mask_sentence = info_tuple[3].replace('[MASK]', ground_truth)
        for morph in nltk.pos_tag(nltk.word_tokenize(no_mask_sentence)):
            if morph[0] == ground_truth and morph[1] in matched_pos_list:
                particular_gt_type_dict[ground_truth] = info_tuple
                break

    return particular_gt_type_dict


def make_datalist_for_graph(relevant_prompt_prob_change_dict, unrelated_prompt_prob_change_dict, *, graph_type: str):
    percentage_list = []
    labels = []
    ground_truth_list = []

    # "c"はグラフを作る上で恣意的に使っている変数なので(全てのエンティティをグラフにしようとすると，600エンティティとかあるので...)，最終的にはちょうどよく設定したい
    c = 0

    if graph_type == "suppress":
        for relevant_prompt_dict_element, unrelated_prompt_dict_element in zip(relevant_prompt_prob_change_dict.items(), unrelated_prompt_prob_change_dict.items()):
            # ground_truthの単語が10文字以上の場合 or 精製されたニューロンの数が0個の場合は，グラフを作成する上で{文字が被ってしまう, 必ず0%になってしまう}不都合が生じるため，暫定的に除外する
            if len(relevant_prompt_dict_element[0]) >= 10 or int(relevant_prompt_dict_element[1][0]) == 0:
                c += 1
                continue
            else:
                # relevant promptの場合の結果の比率を計算し，dataframeを作るためのリストに代入
                percentage_list.append((float(relevant_prompt_dict_element[1][2]) - float(relevant_prompt_dict_element[1][1])) / float(relevant_prompt_dict_element[1][1]) * 100)
                labels.append("活性値を抑制し，知識ニューロンに紐づくプロンプトを予測")
                ground_truth_list.append(relevant_prompt_dict_element[0])

                # unrelated promptの場合の結果の比率を計算し，dataframeを作るためのリストに代入
                percentage_list.append((float(unrelated_prompt_dict_element[1][2]) - float(unrelated_prompt_dict_element[1][1])) / float(unrelated_prompt_dict_element[1][1]) * 100)
                labels.append("活性値を抑制し，知識ニューロンとは無関係のプロンプトを予測")
                ground_truth_list.append(unrelated_prompt_dict_element[0])

                c += 1
    elif graph_type == "enhance":
        for relevant_prompt_dict_element, unrelated_prompt_dict_element in zip(relevant_prompt_prob_change_dict.items(), unrelated_prompt_prob_change_dict.items()):
            if len(relevant_prompt_dict_element[0]) >= 10 or int(relevant_prompt_dict_element[1][0]) == 0:
                c += 1
                continue
            else:
                # relevant promptの場合の結果の比率を計算し，dataframeを作るためのリストに代入
                percentage_list.append((float(relevant_prompt_dict_element[1][2]) - float(relevant_prompt_dict_element[1][1])) / float(relevant_prompt_dict_element[1][1]) * 100)
                labels.append("活性値を増幅し，知識ニューロンに紐づくプロンプトを予測")
                ground_truth_list.append(relevant_prompt_dict_element[0])

                # unrelated promptの場合の結果の比率を計算し，dataframeを作るためのリストに代入
                percentage_list.append((float(unrelated_prompt_dict_element[1][2]) - float(unrelated_prompt_dict_element[1][1])) / float(unrelated_prompt_dict_element[1][1]) * 100)
                labels.append("活性値を増幅し，知識ニューロンとは無関係のプロンプトを予測")
                ground_truth_list.append(unrelated_prompt_dict_element[0])

                c += 1

    return percentage_list, labels, ground_truth_list


def get_datalist(relevant_prompt_result_path, unrelated_prompt_result_path, ground_truth_type, *, graph_type):
    if ground_truth_type == "all":
        relevant_prompt_prob_change_dict = extract_dict_from_result_file(relevant_prompt_result_path)
        unrelated_prompt_prob_change_dict = extract_dict_from_result_file(unrelated_prompt_result_path)
    else:
        relevant_prompt_prob_change_dict = extract_gt_type_dict(extract_dict_from_result_file(relevant_prompt_result_path), ground_truth_type)
        unrelated_prompt_prob_change_dict = extract_gt_type_dict(extract_dict_from_result_file(unrelated_prompt_result_path), ground_truth_type)

    return make_datalist_for_graph(relevant_prompt_prob_change_dict, unrelated_prompt_prob_change_dict, graph_type=graph_type)


def make_suppress_graph(relevant_prompt_result_path, unrelated_prompt_result_path, root_save_path, ground_truth_type, entity_index):
    percentage_list, labels, ground_truth_list = get_datalist(relevant_prompt_result_path, unrelated_prompt_result_path, ground_truth_type, graph_type="suppress")

    number_of_entities_show_in_x_axis = 15
    start_entity_index = entity_index

    df = pd.DataFrame({"正解を選ぶ確率の減少率[%]": percentage_list[2*start_entity_index:2*start_entity_index+2*number_of_entities_show_in_x_axis],
                       "凡例": labels[2*start_entity_index:2*start_entity_index+2*number_of_entities_show_in_x_axis],
                       "概念": ground_truth_list[2*start_entity_index:2*start_entity_index+2*number_of_entities_show_in_x_axis]})

    sns_settings()
    fig, ax = plt.subplots(1, 1, figsize=(40, 10), tight_layout=True)

    graph = sns.barplot(x="概念", y='正解を選ぶ確率の減少率[%]', data=df, hue="凡例", ax=ax, palette=sns.light_palette("blue", n_colors=3, reverse=True))
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=32)  # 凡例の位置を調整する
    figure = graph.get_figure()

    save_path = os.path.join(root_save_path, f"{ground_truth_type}_suppressed_graph.png")
    figure.savefig(save_path)
    print(f"figure is saved in {save_path}")
    plt.close()


def make_enhance_graph(relevant_prompt_result_path, unrelated_prompt_result_path, root_save_path, ground_truth_type, entity_index):
    percentage_list, labels, ground_truth_list = get_datalist(relevant_prompt_result_path, unrelated_prompt_result_path, ground_truth_type, graph_type="enhance")

    number_of_entities_show_in_x_axis = 15
    start_entity_index = entity_index

    df = pd.DataFrame({"正解を選ぶ確率の増加率[%]": percentage_list[2*start_entity_index:2*start_entity_index+2*number_of_entities_show_in_x_axis],
                       "凡例": labels[2*start_entity_index:2*start_entity_index+2*number_of_entities_show_in_x_axis],
                       "概念": ground_truth_list[2*start_entity_index:2*start_entity_index+2*number_of_entities_show_in_x_axis]})

    sns_settings()
    fig, ax = plt.subplots(1, 1, figsize=(40, 10), tight_layout=True)

    graph = sns.barplot(x="概念", y='正解を選ぶ確率の増加率[%]', data=df, hue="凡例", ax=ax, palette=sns.light_palette("red", n_colors=3, reverse=True))
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=32)
    figure = graph.get_figure()

    save_path = os.path.join(root_save_path, f"{ground_truth_type}_enhanced_graph.png")
    figure.savefig(save_path)
    print(f"figure is saved in {save_path}")
    plt.close()


def make_dataframe_for_histogram(percentage_list, labels, ground_truth_list, *, graph_type, prompt_type):
    if prompt_type == "relevant":
        matched_tail_string = "紐づくプロンプトを予測"
    elif prompt_type == "unrelated":
        matched_tail_string = "無関係のプロンプトを予測"
    else:
        assert False, "need to debug, around 'make_dataframe_for_histogram, prompt_type'"

    if graph_type == "suppress":
        axis_titlehead = "減少"
    elif graph_type == "enhance":
        axis_titlehead = "増加"
    else:
        assert False, "need to debug, around 'make_dataframe_for_histogram, graph_type'"

    prompt_percentage_list = []
    prompt_labels = []
    prompt_ground_truth_list = []
    for percentage, label, ground_truth in zip(percentage_list, labels, ground_truth_list):
        if label.endswith(matched_tail_string):
            prompt_percentage_list.append(percentage)
            prompt_labels.append(label)
            prompt_ground_truth_list.append(ground_truth)

    df = pd.DataFrame({f"正解を選ぶ確率の{axis_titlehead}率[%]": prompt_percentage_list,
                       "凡例": prompt_labels,
                       "概念": prompt_ground_truth_list})

    return df


def make_suppress_histogram(relevant_prompt_result_path, unrelated_prompt_result_path, root_save_path, ground_truth_type):
    percentage_list, labels, ground_truth_list = get_datalist(relevant_prompt_result_path, unrelated_prompt_result_path, ground_truth_type, graph_type="suppress")

    df_relevant = make_dataframe_for_histogram(percentage_list, labels, ground_truth_list, graph_type="suppress", prompt_type="relevant")
    df_unrelated = make_dataframe_for_histogram(percentage_list, labels, ground_truth_list, graph_type="suppress", prompt_type="unrelated")

    sns_settings()
    fig, ax = plt.subplots(1, 2, figsize=(40, 20), tight_layout=True)

    sturges = lambda n: math.ceil(math.log2(n*2))
    bins = sturges(len(df_relevant["Decreasing Ratio of Correct Probability[%]"]))

    sns.histplot(data=df_relevant["Decreasing Ratio of Correct Probability[%]"], bins=50, ax=ax[0], color="b")
    ax[0].set(title="Relevant prompts")
    sns.histplot(data=df_unrelated["Decreasing Ratio of Correct Probability[%]"], bins=50, ax=ax[1], color="g")
    ax[1].set(title="Unrelated prompts")
    fig.suptitle(f"The decreasing ratio of the probability after suppressing the activation ({ground_truth_type})")
    figure = fig.get_figure()

    save_path = os.path.join(root_save_path, f"{ground_truth_type}_suppressed_histogram.png")
    figure.savefig(save_path)
    print(f"figure is saved in {save_path}")
    plt.close()


def make_enhance_histogram(relevant_prompt_result_path, unrelated_prompt_result_path, root_save_path, ground_truth_type):
    percentage_list, labels, ground_truth_list = get_datalist(relevant_prompt_result_path, unrelated_prompt_result_path, ground_truth_type, graph_type="enhance")

    df_relevant = make_dataframe_for_histogram(percentage_list, labels, ground_truth_list, graph_type="enhance", prompt_type="relevant")
    df_unrelated = make_dataframe_for_histogram(percentage_list, labels, ground_truth_list, graph_type="enhance", prompt_type="unrelated")

    sns_settings()
    fig, ax = plt.subplots(1, 2, figsize=(40, 20), tight_layout=True)

    sturges = lambda n: math.ceil(math.log2(n*2))
    bins = sturges(len(df_relevant["Increasing Ratio of Correct Probability[%]"]))

    sns.histplot(data=df_relevant["Increasing Ratio of Correct Probability[%]"], bins=50, ax=ax[0], color="r")
    ax[0].set(title="Relevant prompts")
    sns.histplot(data=df_unrelated["Increasing Ratio of Correct Probability[%]"], bins=50, ax=ax[1], color="y")
    ax[1].set(title="Unrelated prompts")
    fig.suptitle(f"The increasing ratio of the probability after enhancing the activation ({ground_truth_type})")
    figure = fig.get_figure()

    save_path = os.path.join(root_save_path, f"{ground_truth_type}_enhanced_histogram.png")
    figure.savefig(save_path)
    print(f"figure is saved in {save_path}")
    plt.close()


def make_suppress_histogram_comparing_prompt_relevance(relevant_prompt_result_path, unrelated_prompt_result_path, root_save_path, ground_truth_type, only_return_dataframes=False):
    percentage_list, labels, ground_truth_list = get_datalist(relevant_prompt_result_path, unrelated_prompt_result_path, ground_truth_type, graph_type="suppress")

    df_relevant = make_dataframe_for_histogram(percentage_list, labels, ground_truth_list, graph_type="suppress", prompt_type="relevant")
    df_unrelated = make_dataframe_for_histogram(percentage_list, labels, ground_truth_list, graph_type="suppress", prompt_type="unrelated")

    if not only_return_dataframes:
        df_suppress = pd.concat([df_relevant, df_unrelated])
        len_df_suppress = len(df_suppress.index)
        df_suppress["new_index"] = [x for x in range(len_df_suppress)]
        df_suppress.set_index("new_index", inplace=True)
        df_suppress = df_suppress.drop("概念", axis=1)

        sns_settings()
        fig, ax = plt.subplots(1, 1, figsize=(30, 15), tight_layout=True)

        sns.histplot(data=df_suppress, x="正解を選ぶ確率の減少率[%]", hue="凡例", bins=28, binrange=(-100, 40), ax=ax)
        figure = fig.get_figure()

        save_path = os.path.join(root_save_path, f"{ground_truth_type}_suppressed_overlapping_histogram.png")
        figure.savefig(save_path)
        print(f"figure is saved in {save_path}")
        plt.close()

    return df_relevant, df_unrelated


def make_enhance_histogram_comparing_prompt_relevance(relevant_prompt_result_path, unrelated_prompt_result_path, root_save_path, ground_truth_type, only_return_dataframes=False):
    percentage_list, labels, ground_truth_list = get_datalist(relevant_prompt_result_path, unrelated_prompt_result_path, ground_truth_type, graph_type="enhance")

    df_relevant = make_dataframe_for_histogram(percentage_list, labels, ground_truth_list, graph_type="enhance", prompt_type="relevant")
    df_unrelated = make_dataframe_for_histogram(percentage_list, labels, ground_truth_list, graph_type="enhance", prompt_type="unrelated")

    if not only_return_dataframes:
        df_enhance = pd.concat([df_relevant, df_unrelated])
        len_df_enhance = len(df_enhance.index)
        df_enhance["new_index"] = [x for x in range(len_df_enhance)]
        df_enhance.set_index("new_index", inplace=True)
        df_enhance = df_enhance.drop("概念", axis=1)

        sns_settings()
        fig, ax = plt.subplots(1, 1, figsize=(30, 15), tight_layout=True)

        sns.histplot(data=df_enhance, x="正解を選ぶ確率の増加率[%]", hue="凡例", bins=44, binrange=(-100, 1000), ax=ax)
        figure = fig.get_figure()

        save_path = os.path.join(root_save_path, f"{ground_truth_type}_enhanced_overlapping_histogram.png")
        figure.savefig(save_path)
        print(f"figure is saved in {save_path}")
        plt.close()

    return df_relevant, df_unrelated


def make_histograms_comparing_entity_and_concept(df_dict, root_save_path, for_article=False):
    # make "suppress_activation_with_relevant_prompts" histogram
    df_entity_suppress_relevant = df_dict["entity_suppress_relevant"].replace({"凡例": {"活性値を抑制し，知識ニューロンに紐づくプロンプトを予測": "名詞"}})
    df_concept_suppress_relevant = df_dict["concept_suppress_relevant"].replace({"凡例": {"活性値を抑制し，知識ニューロンに紐づくプロンプトを予測": "動詞・形容詞・副詞"}})

    df_suppress_relevant = pd.concat([df_entity_suppress_relevant, df_concept_suppress_relevant])
    len_df_suppress_relevant = len(df_suppress_relevant.index)
    df_suppress_relevant["new_index"] = [x for x in range(len_df_suppress_relevant)]
    df_suppress_relevant.set_index("new_index", inplace=True)
    df_suppress_relevant = df_suppress_relevant.drop("概念", axis=1)

    sns_settings2()
    fig, ax = plt.subplots(1, 1, figsize=(24, 10), tight_layout=True)

    sns.histplot(data=df_suppress_relevant, x="正解を選ぶ確率の減少率[%]", hue="凡例", bins=28, binrange=(-100, 40), ax=ax)
    figure = fig.get_figure()

    save_path = os.path.join(root_save_path, f"relevant_suppressed_histogram.png")
    figure.savefig(save_path)
    print(f"figure is saved in {save_path}")
    plt.close()

    # make "enhance_activation_with_relevant_prompts" histogram
    df_entity_enhance_relevant = df_dict["entity_enhance_relevant"].replace({"凡例": {"活性値を増幅し，知識ニューロンに紐づくプロンプトを予測": "名詞"}})
    df_concept_enhance_relevant = df_dict["concept_enhance_relevant"].replace({"凡例": {"活性値を増幅し，知識ニューロンに紐づくプロンプトを予測": "動詞・形容詞・副詞"}})

    df_enhance_relevant = pd.concat([df_entity_enhance_relevant, df_concept_enhance_relevant])
    len_df_enhance_relevant = len(df_enhance_relevant.index)
    df_enhance_relevant["new_index"] = [x for x in range(len_df_enhance_relevant)]
    df_enhance_relevant.set_index("new_index", inplace=True)
    df_enhance_relevant = df_enhance_relevant.drop("概念", axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(24, 10), tight_layout=True)

    sns.histplot(data=df_enhance_relevant, x="正解を選ぶ確率の増加率[%]", hue="凡例", bins=44, binrange=(-100, 1000), ax=ax)
    figure = fig.get_figure()

    save_path = os.path.join(root_save_path, f"relevant_enhanced_histogram.png")
    figure.savefig(save_path)
    print(f"figure is saved in {save_path}")
    plt.close()

    if not for_article:
        # make "suppress_activation_with_unrelated_prompts" histogram
        df_entity_suppress_unrelated = df_dict["entity_suppress_unrelated"].replace({"凡例": {"活性値を抑制し，知識ニューロンとは無関係のプロンプトを予測": "名詞"}})
        df_concept_suppress_unrelated = df_dict["concept_suppress_unrelated"].replace({"凡例": {"活性値を抑制し，知識ニューロンとは無関係のプロンプトを予測": "動詞・形容詞・副詞"}})

        df_suppress_unrelated = pd.concat([df_entity_suppress_unrelated, df_concept_suppress_unrelated])
        len_df_suppress_unrelated = len(df_suppress_unrelated.index)
        df_suppress_unrelated["new_index"] = [x for x in range(len_df_suppress_unrelated)]
        df_suppress_unrelated.set_index("new_index", inplace=True)
        df_suppress_unrelated = df_suppress_unrelated.drop("概念", axis=1)

        fig, ax = plt.subplots(1, 1, figsize=(24, 10), tight_layout=True)

        sns.histplot(data=df_suppress_unrelated, x="正解を選ぶ確率の減少率[%]", hue="凡例", bins=24, binrange=(-50, 10), ax=ax)
        figure = fig.get_figure()

        save_path = os.path.join(root_save_path, f"unrelated_suppressed_histogram.png")
        figure.savefig(save_path)
        print(f"figure is saved in {save_path}")
        plt.close()

        # make "enhance_activation_with_unrelated_prompts" histogram
        df_entity_enhance_unrelated = df_dict["entity_enhance_unrelated"].replace({"凡例": {"活性値を増幅し，知識ニューロンとは無関係のプロンプトを予測": "名詞"}})
        df_concept_enhance_unrelated = df_dict["concept_enhance_unrelated"].replace({"凡例": {"活性値を増幅し，知識ニューロンとは無関係のプロンプトを予測": "動詞・形容詞・副詞"}})

        df_enhance_unrelated = pd.concat([df_entity_enhance_unrelated, df_concept_enhance_unrelated])
        len_df_enhance_unrelated = len(df_enhance_unrelated.index)
        df_enhance_unrelated["new_index"] = [x for x in range(len_df_enhance_unrelated)]
        df_enhance_unrelated.set_index("new_index", inplace=True)
        df_enhance_unrelated = df_enhance_unrelated.drop("概念", axis=1)

        fig, ax = plt.subplots(1, 1, figsize=(24, 10), tight_layout=True)

        sns.histplot(data=df_enhance_unrelated, x="正解を選ぶ確率の増加率[%]", hue="凡例", bins=28, binrange=(-15, 20), ax=ax)
        figure = fig.get_figure()

        save_path = os.path.join(root_save_path, f"unrelated_enhanced_histogram.png")
        figure.savefig(save_path)
        print(f"figure is saved in {save_path}")
        plt.close()


def main():
    args = make_parser()

    if args.dataset_type == "TREx" and args.entity_type == "subject":
        try:
            raise NotImplementedError("didn't conduct the experiment with such a condition.")
        except NotImplementedError:
            traceback.print_exc()
            sys.exit(1)

    """
    if args.graph_type == "other" and 0 <= args.for_any_entity and args.for_any_entity <= 729:
        entity_index = args.for_any_entity
    elif args.graph_type == "kitten" and args.for_any_entity == 100000:
        entity_index = 126
    elif args.graph_type == "pos" and args.for_any_entity == 100000:
        entity_index = 50
    else:
        try:
            raise NotImplementedError("please set '--graph_type' or '--for_any_entity' argument correctly")
        except NotImplementedError:
            traceback.print_exc()
            sys.exit(1)
    """
    if not args.article_figures:
        entity_index = args.for_any_entity

    result_path = os.path.join("work", "result", args.dataset_type, args.entity_type)
    suppress_relevant = os.path.join(result_path, "suppress_activation_and_relevant_prompts.txt")
    suppress_unrelated = os.path.join(result_path, "suppress_activation_and_unrelated_prompts.txt")
    enhance_relevant = os.path.join(result_path, "enhance_activation_and_relevant_prompts.txt")
    enhance_unrelated = os.path.join(result_path, "enhance_activation_and_unrelated_prompts.txt")

    df_dict_for_comparing_entity_and_concept = defaultdict(int)

    if args.article_figures:
        root_path_of_saving_graph = os.path.join("work", "figure", "article", args.dataset_type, args.entity_type)
        os.makedirs(root_path_of_saving_graph, exist_ok=True)

        # figure 3, 4, 5, 6
        make_suppress_graph(suppress_relevant, suppress_unrelated, root_path_of_saving_graph, "all", 126)
        make_enhance_graph(enhance_relevant, enhance_unrelated, root_path_of_saving_graph, "all", 126)
        hoge, fuga = make_suppress_histogram_comparing_prompt_relevance(suppress_relevant, suppress_unrelated, root_path_of_saving_graph, "all")
        foo, bar = make_enhance_histogram_comparing_prompt_relevance(enhance_relevant, enhance_unrelated, root_path_of_saving_graph, "all")

        # preparation for figure 7, 8
        df_suppress_relevant, df_suppress_unrelated = make_suppress_histogram_comparing_prompt_relevance(suppress_relevant, suppress_unrelated, root_path_of_saving_graph, "entity", only_return_dataframes=True)
        df_enhance_relevant, df_enhance_unrelated = make_enhance_histogram_comparing_prompt_relevance(enhance_relevant, enhance_unrelated, root_path_of_saving_graph, "entity", only_return_dataframes=True)
        df_dict_for_comparing_entity_and_concept["entity_suppress_relevant"] = df_suppress_relevant
        df_dict_for_comparing_entity_and_concept["entity_suppress_unrelated"] = df_suppress_unrelated
        df_dict_for_comparing_entity_and_concept["entity_enhance_relevant"] = df_enhance_relevant
        df_dict_for_comparing_entity_and_concept["entity_enhance_unrelated"] = df_enhance_unrelated

        df_suppress_relevant, df_suppress_unrelated = make_suppress_histogram_comparing_prompt_relevance(suppress_relevant, suppress_unrelated, root_path_of_saving_graph, "concept", only_return_dataframes=True)
        df_enhance_relevant, df_enhance_unrelated = make_enhance_histogram_comparing_prompt_relevance(enhance_relevant, enhance_unrelated, root_path_of_saving_graph, "concept", only_return_dataframes=True)
        df_dict_for_comparing_entity_and_concept["concept_suppress_relevant"] = df_suppress_relevant
        df_dict_for_comparing_entity_and_concept["concept_suppress_unrelated"] = df_suppress_unrelated
        df_dict_for_comparing_entity_and_concept["concept_enhance_relevant"] = df_enhance_relevant
        df_dict_for_comparing_entity_and_concept["concept_enhance_unrelated"] = df_enhance_unrelated

        # figure 7, 8
        make_histograms_comparing_entity_and_concept(df_dict_for_comparing_entity_and_concept, root_path_of_saving_graph, for_article=True)

    else:
        root_path_of_saving_graph = os.path.join("work", "figure", args.dataset_type, args.entity_type)
        os.makedirs(root_path_of_saving_graph, exist_ok=True)

        for ground_truth_type in ["all", "entity", "concept"]:
            make_suppress_graph(suppress_relevant, suppress_unrelated, root_path_of_saving_graph, ground_truth_type, entity_index)
            make_enhance_graph(enhance_relevant, enhance_unrelated, root_path_of_saving_graph, ground_truth_type, entity_index)
            #make_suppress_histogram(suppress_relevant, suppress_unrelated, root_path_of_saving_graph, ground_truth_type)
            #make_enhance_histogram(enhance_relevant, enhance_unrelated, root_path_of_saving_graph, ground_truth_type)
            df_suppress_relevant, df_suppress_unrelated = make_suppress_histogram_comparing_prompt_relevance(suppress_relevant, suppress_unrelated, root_path_of_saving_graph, ground_truth_type)
            df_enhance_relevant, df_enhance_unrelated = make_enhance_histogram_comparing_prompt_relevance(enhance_relevant, enhance_unrelated, root_path_of_saving_graph, ground_truth_type)
            if ground_truth_type == "entity":
                df_dict_for_comparing_entity_and_concept["entity_suppress_relevant"] = df_suppress_relevant
                df_dict_for_comparing_entity_and_concept["entity_suppress_unrelated"] = df_suppress_unrelated
                df_dict_for_comparing_entity_and_concept["entity_enhance_relevant"] = df_enhance_relevant
                df_dict_for_comparing_entity_and_concept["entity_enhance_unrelated"] = df_enhance_unrelated
            elif ground_truth_type == "concept":
                df_dict_for_comparing_entity_and_concept["concept_suppress_relevant"] = df_suppress_relevant
                df_dict_for_comparing_entity_and_concept["concept_suppress_unrelated"] = df_suppress_unrelated
                df_dict_for_comparing_entity_and_concept["concept_enhance_relevant"] = df_enhance_relevant
                df_dict_for_comparing_entity_and_concept["concept_enhance_unrelated"] = df_enhance_unrelated

        make_histograms_comparing_entity_and_concept(df_dict_for_comparing_entity_and_concept, root_path_of_saving_graph)


if __name__ == '__main__':
    main()
