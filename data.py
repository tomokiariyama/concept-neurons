from tqdm import tqdm
from collections import defaultdict
import json
from logzero import logger
import sys
import re
import unicodedata


def extract_dicts_from_jsonlines(file_path):
    """
    形式の異なるデータセット(TREx, ConceptNet)から，'subject, object, template'をキーとして持つ辞書を返す関数
    """

    with open(file_path) as fi:
        dataset_list = []
        for line in tqdm(fi):
            d = defaultdict(str)

            case = json.loads(line)
            try:
                d["sub"] = case["sub_surface"]
            except KeyError:
                try:
                    d["sub"] = case["sub_label"]
                except KeyError:
                    try:
                        d["sub"] = case["sub"]
                    except KeyError:
                        logger.info(f"case: {case}")
                        logger.error("There is no key corresponding to 'subject' in this dataset.")
                        sys.exit(1)

            try:
                d["obj"] = case["obj_surface"]
            except KeyError:
                try:
                    d["obj"] = case["obj_label"]
                except KeyError:
                    try:
                        d["obj"] = case["obj"]
                    except KeyError:
                        logger.info(f"case: {case}")
                        logger.error("There is no key corresponding to 'object' in this dataset.")
                        sys.exit(1)

            try:
                for masked_sentence in case["masked_sentences"]:
                    if masked_sentence.count("[MASK]") == 1:
                        d["masked_sentence"] = masked_sentence
                # [MASK]トークンを一つのみ含むmasked_sentenceが見つからなかった場合は，そのcaseはskipする
                if not d["masked_sentence"]:
                    continue
            except KeyError:
                try:
                    d["masked_sentence"] = case["evidences"][0]["masked_sentence"]
                except KeyError:
                    logger.info(f"case: {case}")
                    logger.error("There is no 'masked_sentence' key in this dataset.")
                    sys.exit(1)

            dataset_list.append(d)

    return dataset_list


def format_data_with_entity_type(entity_type, dataset_list, num_of_templates, max_words):
    if entity_type == "subject":
        d = defaultdict(set)

        # まずは各subject entityごとのテンプレートにおいて，[MASK]をobjに，subを[MASK]に置き換える
        # その上で，subをキーとした辞書にset型でテンプレートを追加していく
        # set型にするのは，同じテンプレートでobjが違うものがあるがテンプレートが同一なら一つとカウントするため
        for case in tqdm(dataset_list):
            try:
                r = re.compile(f'{case["sub"]}', re.IGNORECASE)  # subの先頭の大文字小文字を考慮せずに[MASK]に置き換える
            except re.error:
                logger.warning(f'skipped a case which has sub_label: {case["sub"]}')
                continue

            # 元のmasked sentenceに，[MASK]トークンと変換すべきsubが完全一致では含まれていない場合があるため，その場合は除く
            if not re.search(r, case["masked_sentence"]):
                logger.warning(f'skipped a case which has masked_sentence with no sub_label, sub_label: {case["sub"]}, masked_sentence: {case["masked_sentence"]}')
                continue

            no_mask_sentence = case["masked_sentence"].replace('[MASK]', case["obj"])
            new_masked_sentence = re.sub(r, '[MASK]', no_mask_sentence,
                                         1)  # 念の為，マスクトークンが文中に一つしか現れないようにしておく（元の文中にsubが2つ以上現れることはないと思うが）
            new_masked_sentence = unicodedata.normalize("NFKD", new_masked_sentence)  # Unicodeのノーブレークスペースなどを置き換える

            if len(new_masked_sentence.split(" ")) <= max_words:
                d[case["sub"]].add(new_masked_sentence)
            else:
                continue

        # 既定のテンプレート数に満たないsubject entityを除く
        delete_entities = []
        for sub in d.keys():
            if len(d[sub]) < num_of_templates:
                delete_entities.append(sub)
        for delete_key in delete_entities:
            del d[delete_key]

        return d
    elif entity_type == "object":
        d = defaultdict(set)

        # 各object entityごとのテンプレートを辞書にset型で登録する
        for case in tqdm(dataset_list):
            if len(case["masked_sentence"].split(" ")) <= max_words:
                d[case["obj"]].add(case["masked_sentence"])
            else:
                continue

        # 既定のテンプレート数に満たないobject entityを除く
        delete_entities = []
        for obj in d.keys():
            if len(d[obj]) < num_of_templates:
                delete_entities.append(obj)
        for delete_key in delete_entities:
            del d[delete_key]

        return d
    else:
        try:
            raise ValueError("entity type is somewhat wrong")
        except ValueError as e:
            print(e)
        sys.exit(1)
