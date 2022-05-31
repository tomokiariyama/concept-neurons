import os
import numpy as np
import torch
from transformers import BertModel, BertTokenizer, PreTrainedTokenizerBase


def main():
    # 必要なデータへのパス
    os.makedirs(os.path.join("work", "predicting_places"), exist_ok=True)

    train_text_path = os.path.join("data", "for_predicting_places", "ConceptNet", "subject", "nt_4_at_0.2_mw_10", "train_place_of_neurons.csv")
    x_train_save_path_tensor = os.path.join("work", "predicting_places", "x_train_tensor.pt")
    y_train_save_path_tensor = os.path.join("work", "predicting_places", "y_train_tensor.pt")

    valid_text_path = os.path.join("data", "for_predicting_places", "ConceptNet", "subject", "nt_4_at_0.2_mw_10", "valid_place_of_neurons.csv")
    x_valid_save_path_tensor = os.path.join("work", "predicting_places", "x_valid_tensor.pt")
    y_valid_save_path_tensor = os.path.join("work", "predicting_places", "y_valid_tensor.pt")

    test_text_path = os.path.join("data", "for_predicting_places", "ConceptNet", "subject", "nt_4_at_0.2_mw_10", "test_place_of_neurons.csv")
    x_test_save_path_tensor = os.path.join("work", "predicting_places", "x_test_tensor.pt")
    y_test_save_path_tensor = os.path.join("work", "predicting_places", "y_test_tensor.pt")

    # モデルとトークナイザ
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    token_embeddings = model.get_input_embeddings().weight.clone()  # 単語埋め込み
    vocab = tokenizer.get_vocab()  # 単語がキー、整数値がバリューになっていて、その整数値がtoken_embeddingsのインデックスに対応する

    # train dataのtensor化
    word_embedding_dim = 768
    number_of_train_set = 560
    with open(train_text_path, "r") as tr_fi:
        x_list = []
        y_list = []
        for line in tr_fi:
            concept = line.split(",")[0]
            layer_num = line.split(",")[1:-1]  # 訓練や推測には使用しない
            median = line.split(",")[-1]

            try:
                print(concept)
                print(tokenizer.convert_tokens_to_ids(concept))
                print(vocab[concept])
                #word_embedding = token_embeddings[int(tokenizer.convert_tokens_to_ids([concept])[0])]
                #word_embedding = token_embeddings[vocab[concept]]
                word_embedding = token_embeddings[tokenizer.convert_tokens_to_ids(concept)]
            except:
                print(f"missing word embedding: '{concept}'")
                continue

            assert len(word_embedding) == word_embedding_dim, f"the shape of word embedding doesn't match with 768, now len(word_embedding) == {len(word_embedding)}."

            x_list.append(word_embedding)
            y_list.append(torch.tensor([float(median)]))  # 文字列medianをfloat → list → tensor へと次々変換

        x_train_tensor = torch.stack(x_list)
        y_train_tensor = torch.stack(y_list)

        assert x_train_tensor == torch.Size([number_of_train_set, word_embedding_dim]), f"for train tensor data's shape doesn't match with torch.Size([560, 768]), now x_train_tensor == {x_train_tensor.shape}"
        assert y_train_tensor == torch.Size([number_of_train_set, 1]), f"for train tensor data for answer's shape doesn't match with torch.Size([560, 1]), now y_train_tensor == {y_train_tensor.shape}"

        torch.save(x_train_tensor, x_train_save_path_tensor)
        torch.save(y_train_tensor, y_train_save_path_tensor)

    # valid dataのtensor化
    number_of_valid_set = 71
    with open(valid_text_path, "r") as va_fi:
        x_list = []
        y_list = []
        for line in va_fi:
            concept = line.split(",")[0]
            layer_num = line.split(",")[1:-1]  # 訓練や推測には使用しない
            median = line.split(",")[-1]

            try:
                word_embedding = token_embeddings[vocab[concept]]
            except:
                print(f"missing word embedding: '{concept}'")
                continue

            assert len(
                word_embedding) == word_embedding_dim, f"the shape of word embedding doesn't match with 768, now len(word_embedding) == {len(word_embedding)}."

            x_list.append(word_embedding)
            y_list.append(torch.tensor([float(median)]))  # 文字列medianをfloat → list → tensor へと次々変換

        x_valid_tensor = torch.stack(x_list)
        y_valid_tensor = torch.stack(y_list)

        assert x_valid_tensor == torch.Size([number_of_valid_set,
                                             word_embedding_dim]), f"for valid tensor data's shape doesn't match with torch.Size([71, 768]), now x_valid_tensor == {x_valid_tensor.shape}"
        assert y_valid_tensor == torch.Size([number_of_valid_set,
                                             1]), f"for valid tensor data for answer's shape doesn't match with torch.Size([71, 1]), now y_valid_tensor == {y_valid_tensor.shape}"

        torch.save(x_valid_tensor, x_valid_save_path_tensor)
        torch.save(y_valid_tensor, y_valid_save_path_tensor)

    # test dataのtensor化
    number_of_test_set = 70
    with open(test_text_path, "r") as te_fi:
        x_list = []
        y_list = []
        for line in te_fi:
            concept = line.split(",")[0]
            layer_num = line.split(",")[1:-1]  # 訓練や推測には使用しない
            median = line.split(",")[-1]

            try:
                word_embedding = token_embeddings[vocab[concept]]
            except:
                print(f"missing word embedding: '{concept}'")
                continue

            assert len(
                word_embedding) == word_embedding_dim, f"the shape of word embedding doesn't match with 768, now len(word_embedding) == {len(word_embedding)}."

            x_list.append(word_embedding)
            y_list.append(torch.tensor([float(median)]))  # 文字列medianをfloat → list → tensor へと次々変換

        x_test_tensor = torch.stack(x_list)
        y_test_tensor = torch.stack(y_list)

        assert x_test_tensor == torch.Size([number_of_test_set,
                                             word_embedding_dim]), f"for test tensor data's shape doesn't match with torch.Size([70, 768]), now x_test_tensor == {x_test_tensor.shape}"
        assert y_test_tensor == torch.Size([number_of_test_set,
                                             1]), f"for test tensor data for answer's shape doesn't match with torch.Size([70, 1]), now y_test_tensor == {y_test_tensor.shape}"

        torch.save(x_test_tensor, x_test_save_path_tensor)
        torch.save(y_test_tensor, y_test_save_path_tensor)


if __name__ == "__main__":
    main()
