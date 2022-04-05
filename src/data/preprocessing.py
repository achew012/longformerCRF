from typing import List, Dict, Any, Tuple
import ipdb
import random
from omegaconf import OmegaConf
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)
from tqdm import tqdm


def convert_char_indices(
    ans_char_start: int,
    ans_char_end: int,
    spans_list: List[List[Tuple[int, int]]],
    max_idx: int,
) -> List:
    # offset has to be List[List[int, int]] or tensor of same shape
    # if char indices more than end idx in last word span, reset indices to 0

    if ans_char_end > max_idx or ans_char_start > max_idx:
        ans_char_start = 0
        ans_char_end = 0

    if ans_char_start == 0 and ans_char_end == 0:
        token_span = [0, 0]
    else:
        token_span = []
        for idx, span in enumerate(spans_list):
            if (
                ans_char_start >= span[0]
                and ans_char_start <= span[1]
                and len(token_span) == 0
            ):
                token_span.append(idx)

            if (
                ans_char_end >= span[0]
                and ans_char_end <= span[1]
                and len(token_span) == 1
            ):
                token_span.append(idx)
                break

        # if token span is incomplete
        if len(token_span) != 2:
            print("cant find token span")
            ipdb.set_trace()

    return token_span


def is_existing_question(natural_question: str, qns_ans: List) -> Tuple[bool, Any]:
    for idx, question_group in enumerate(qns_ans):
        if natural_question in question_group[0]:
            return (True, idx)
    return (False, -1)


def extract_answers(template: Dict) -> List:
    classes = list(template.keys())
    ans = []
    for class_name in classes:
        mentions = template[class_name]
        if len(mentions) > 0:
            for answers in mentions:
                for answer in answers:
                    mention = answer[0]
                    start_idx = answer[1]
                    end_idx = start_idx + len(mention)
        else:
            mention = ""
            start_idx = 0
            end_idx = 0

        if start_idx == 0 and end_idx == 0:
            # if it's a blank answer, 20% chance of being included into the training set
            continue

        ans.append((start_idx, end_idx, mention, class_name))
    return ans


def main_processing(
    doc: Dict, tokenizer: Any, cfg: Any
) -> Tuple:

    docid = doc["docid"]
    context = doc["doctext"]
    tokens = tokenizer.tokenize(context)

    labels = [int(cfg.role_map["O"])]*(cfg.max_input_len)
    length_of_sequence = len(tokens) if len(
        tokens) <= len(labels) else len(labels)

    labels[:length_of_sequence] = [int(cfg.role_map["O"])]*length_of_sequence

    context_encodings = tokenizer(context, padding="max_length", truncation=True,
                                  max_length=cfg.max_input_len, return_offsets_mapping=True, return_tensors="pt")

    for template in doc["templates"]:
        incident = template.pop('incident_type', None)

        answers = extract_answers(template)

        # convert_character_spans_to_word_spans
        for ans_char_start, ans_char_end, mention, class_name in answers:
            sequence_ids = context_encodings.sequence_ids()
            pad_start_idx = sequence_ids[sequence_ids.index(0):].index(None)
            offsets_wo_pad = context_encodings["offset_mapping"][0][sequence_ids.index(
                0):pad_start_idx]

            if ans_char_end > offsets_wo_pad[-1][1] or ans_char_start > offsets_wo_pad[-1][1]:
                ans_char_start = 0
                ans_char_end = 0

            if ans_char_start == 0 and ans_char_end == 0:
                token_span = [0, 0]
            else:
                token_span = []
                for idx, span in enumerate(offsets_wo_pad):
                    if ans_char_start >= span[0] and ans_char_start <= span[1] and len(token_span) == 0:
                        token_span.append(idx)

                    if ans_char_end >= span[0] and ans_char_end <= span[1] and len(token_span) == 1:
                        token_span.append(idx)
                        break

            if token_span != [0, 0]:
                labels[token_span[0]:token_span[0] +
                       1] = [int(cfg.role_map["B-"+class_name])]
                if len(labels[token_span[0]+1:token_span[1]+1]) > 0 and (token_span[1]-token_span[0]) > 0:
                    labels[token_span[0]+1:token_span[1] +
                           1] = [int(cfg.role_map["I-"+class_name])]*(token_span[1]-token_span[0])

    return docid, context, context_encodings, labels


def process_train_data(dataset: List[Dict], tokenizer: Any, cfg: Any) -> List:

    processed_dataset = []
    dataset = [doc for doc in dataset if len(doc["templates"]) > 0]
    for doc in dataset:
        processed_sample = {}
        docid, context, context_encodings, labels = main_processing(
            doc, tokenizer, cfg)

        processed_sample["docid"] = docid
        processed_sample["context"] = context
        processed_sample["input_ids"] = context_encodings["input_ids"].squeeze(
            0)
        processed_sample["attention_mask"] = context_encodings["attention_mask"].squeeze(
            0)
        processed_sample["labels"] = torch.tensor(labels)

        processed_dataset.append(processed_sample)
    return processed_dataset


def process_inference_data(dataset: List[Dict], tokenizer: Any, cfg: Any) -> List:

    processed_dataset = []
    dataset = [doc for doc in dataset if len(doc["templates"]) > 0]
    for doc in dataset:
        docid = doc["docid"]
        context = doc["doctext"]
        context_encodings = tokenizer(context, padding="max_length", truncation=True,
                                      max_length=cfg.max_input_len, return_offsets_mapping=True, return_tensors="pt")

        processed_sample = {}
        processed_sample["docid"] = docid
        processed_sample["context"] = context
        processed_sample["input_ids"] = context_encodings["input_ids"].squeeze(
            0)
        processed_sample["attention_mask"] = context_encodings["attention_mask"].squeeze(
            0)

        processed_dataset.append(processed_sample)
    return processed_dataset
