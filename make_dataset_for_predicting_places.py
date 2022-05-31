import os


def calc_median(layer_num_list):
    try:
        return str(sum(layer_num_list) / len(layer_num_list)) + "\n"
    except:
        return "\n"


def main():
    data_path = os.path.join("work", "result", "ConceptNet", "subject", "nt_4_at_0.2_mw_10", "place_of_neurons.csv")
    os.makedirs(os.path.join("data", "for_predicting_places", "ConceptNet", "subject", "nt_4_at_0.2_mw_10"), exist_ok=True)
    save_path = os.path.join("data", "for_predicting_places", "ConceptNet", "subject", "nt_4_at_0.2_mw_10", "place_of_neurons.csv")

    with open(data_path, "r") as da_fi, open(save_path, "w") as sa_fi:
        # sa_fiには、`<concept>,<layer num>,...,<median of layer nums>`の形で記録する
        sa_fi.write("<LEGEND>: <concept>,<layer num>,...,<median of layer nums>")
        #concept_with_no_knowledge_neuron_flag = False  # 知識ニューロンが見つからなかった概念を表すフラグ
        layer_num_list = []  # その概念の層のナンバーを格納しておくリスト。中央値の計算に用いるので、中身はint型。

        for line in da_fi:
            if not line.startswith("<LEGEND>"):
                if line.startswith("#"):
                    sa_fi.write(calc_median(layer_num_list))
                    #if not concept_with_no_knowledge_neuron_flag:
                    sa_fi.write(line.lstrip("#").rstrip("\n") + ",")
                    layer_num_list = []
                        #concept_with_no_knowledge_neuron_flag = True
                    #elif concept_with_no_knowledge_neuron_flag:
                        #sa_fi.write("\n" + line.lstrip("#") + ",")
                else:
                    layer_num = line.split(",")[0]
                    sa_fi.write(layer_num + ",")
                    layer_num_list.append(int(layer_num))


if __name__=="__main__":
    main()
