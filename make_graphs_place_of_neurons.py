import argparse
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import japanize_matplotlib


# ヒストグラムの書式設定
plt.style.use('ggplot')
font = {'family' : 'meiryo'}
matplotlib.rc('font', **font)


def sns_settings2():
    sns.set(
        context="paper",
        style="whitegrid",
        palette=sns.dark_palette("palegreen", n_colors=2, reverse=True),
        font_scale=4,
        rc={"lines.linewidth": 6, 'grid.linestyle': '--'},
        font='IPAexGothic'
        )


def make_parser():
    parser = argparse.ArgumentParser(description='Conduct experiments')

    parser.add_argument('-dt', '--dataset_type', help="designate the dataset", default="ConceptNet",
                        choices=["ConceptNet", "TREx"])
    parser.add_argument('-et', '--entity_type', help="designate entity type which is masked", default="subject",
                        choices=["subject", "object"])
    parser.add_argument('-nt', '--number_of_templates',
                        help='the minimum number of templates which each entity have. default=4', type=int, default=4)
    parser.add_argument('-at', '--adaptive_threshold', help="the threshold value", type=float, default=0.2)
    parser.add_argument('-mw', '--max_words', help="the maximum number of words which each template can have", type=int,
                        default=10)

    return parser.parse_args()


def extract_csv_row(input_csv_file, output_csv_file):
    """
    作成した結果のcsvファイルから、ニューロンの場所のみを抽出する
    """
    with open(input_csv_file, mode="r") as in_fi, open(output_csv_file, mode="w") as out_fi:
        for line in in_fi:
            if not line.startswith("#") and not line.startswith("<"):
                out_fi.write(line + "\n")


def main():
    args = make_parser()

    result_path = os.path.join("work", "result", args.dataset_type, args.entity_type,
                               f"nt_{args.number_of_templates}_at_{args.adaptive_threshold}_mw_{args.max_words}")

    # 作成した結果のcsvファイルから、ニューロンの場所のみを抽出する
    for kind in ["_", "_noun_", "_verb_adjective_adverb_", "_others_pos_"]:
        input_csv_file = os.path.join(result_path, f"place_of{kind}neurons.csv")
        output_csv_file = os.path.join(result_path, f"only_places_of{kind}neurons.csv")
        extract_csv_row(input_csv_file, output_csv_file)

    # ヒストグラムの作成
    dict_of_df = dict()  # seabornで名詞と動詞・形容詞・副詞を比較するためのdfデータを格納する辞書
    for kind in ["_", "_noun_", "_verb_adjective_adverb_", "_others_pos_"]:
        csv_file = os.path.join(result_path, f"only_places_of{kind}neurons.csv")
        save_path = os.path.join("work", "figure", "ConceptNet", "subject")

        # pandasを使ったヒストグラムの作成（Pyplotインターフェース流儀）
        df = pd.read_csv(csv_file, names=('layer_index', 'index'))
        print(df)
        dict_of_df[f"{kind.strip('_')}"] = df
        df.plot.hist(y=['layer_index'], bins=12, range=(0, 12), alpha=0.5, figsize=(12, 5))
        plt.title(f'Histogram of {kind.strip("_")} concept neurons')
        plt.xlabel('layer index')
        plt.ylabel('freq.')
        plt.savefig(os.path.join(save_path, f"histogram_layer_of{kind}neurons.png"))

    # seabornを使ったヒストグラムの作成（オブジェクト指向インターフェース流儀）
    # seaborn用のpandas dataframeの作成
    # 名詞と動詞等のdfに、各品詞の列を追加する
    dict_of_df["noun"]["pos"] = "noun"
    dict_of_df["verb_adjective_adverb"]["pos"] = "verb, adjective, adverb"

    # 名詞と動詞等のdfを行方向に結合
    df_contain_noun_and_verbs = pd.concat([dict_of_df["noun"], dict_of_df["verb_adjective_adverb"]], axis=0)
    # 行名であるインデックスが結合により被ってしまうので、リネームする
    df_contain_noun_and_verbs.set_axis([i for i in range(len(df_contain_noun_and_verbs))], axis=0, inplace=True)

    sns_settings2()
    fig, ax = plt.subplots(1, 1, figsize=(24, 10), tight_layout=True)

    sns.histplot(data=df_contain_noun_and_verbs, x="layer_index", hue="pos", bins=12, binrange=(0, 12), ax=ax)
    ax.set_title("Comparing the place of noun and verbs' concept neurons")
    figure = fig.get_figure()

    save_path = os.path.join(save_path, f"place_of_layer_of_noun_and_verbs_overlapping_histogram.png")
    figure.savefig(save_path)
    print(f"figure is saved in {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
