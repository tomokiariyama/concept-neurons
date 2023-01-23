from tqdm import tqdm
from collections import defaultdict
import json
from logzero import logger
import sys
import re
import unicodedata
from transformers import BertTokenizer


def extract_raw_dataset_from_jsonlines(file_path):
    """
    The function which returns the dictionary made from {TREx or ConceptNet} dataset and whose keys are 'subject, object, template'.
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
                # If we couldn't find the masked_sentence that has only one [MASK] token, skip that case.
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


def extract_matched_dataset(dataset_list, entity_type, num_of_templates, max_words, is_remove_unk_concept):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    unk_id = tokenizer.convert_tokens_to_ids("UNK")

    if entity_type == "subject":
        d = defaultdict(set)  # key: concept(str), value: set consists of concept's templates

        # At first, replace "[MASK]" to obj, sub to "[MASK]" in the template.
        # Then, append templates as set type to the dictionary whose key is sub.
        # The reason for using the set type is that some templates are the same but have different obj, but if the template is the same, it is counted as one.
        for case in tqdm(dataset_list):
            try:
                r = re.compile(f'{case["sub"]}', re.IGNORECASE)  # Replace sub to "[MASK]" without considering the case of the first letter of sub.
            except re.error:
                logger.warning(f'skipped a case which has sub_label: {case["sub"]}')
                continue

            # Since the original masked_sentence may not contain an exact match between the [MASK] token and the sub to be converted, this case is excluded.
            if not re.search(r, case["masked_sentence"]):
                logger.warning(f'skipped a case which has masked_sentence with no sub_label, sub_label: {case["sub"]}, masked_sentence: {case["masked_sentence"]}')
                continue

            no_mask_sentence = case["masked_sentence"].replace('[MASK]', case["obj"])
            new_masked_sentence = re.sub(r, '[MASK]', no_mask_sentence, 1)  # Make sure that only one mask token appears in a new_masked_sentence.
            new_masked_sentence = unicodedata.normalize("NFKD", new_masked_sentence)  # Replace the Unicode's no-break-space and so on.

            # Restrict the number of maximum words in a template.
            if len(new_masked_sentence.split(" ")) <= max_words:
                d[case["sub"]].add(new_masked_sentence)
            else:
                continue

        # Exclude subject entities that do not meet the default number of templates.
        delete_entities = []
        if is_remove_unk_concept:
            for sub in d.keys():
                if len(d[sub]) < num_of_templates or tokenizer.convert_tokens_to_ids(sub) == unk_id:
                    delete_entities.append(sub)
        else:
            for sub in d.keys():
                if len(d[sub]) < num_of_templates:
                    delete_entities.append(sub)
        for delete_key in delete_entities:
            del d[delete_key]

        return d

    elif entity_type == "object":
        d = defaultdict(set)

        # Register a template for each object entity in the dictionary as a set type
        for case in tqdm(dataset_list):
            # Restrict the number of maximum words in a template.
            if len(case["masked_sentence"].split(" ")) <= max_words:
                d[case["obj"]].add(case["masked_sentence"])
            else:
                continue

        # Exclude object entities that do not meet the default number of templates.
        delete_entities = []
        if is_remove_unk_concept:
            for obj in d.keys():
                if len(d[obj]) < num_of_templates or tokenizer.convert_tokens_to_ids(obj) == unk_id:
                    delete_entities.append(obj)
        else:
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
