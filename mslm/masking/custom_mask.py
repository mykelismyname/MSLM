# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 05/07/2021 
# @Contact: michealabaho265@gmail.com
import re
from datasets import Dataset
import torch
from typing import List
'''
    Masking specific tokens of the input dataset
'''

def customMask(tokenized_input, tokenizer, labels_list, mask_id, custom_mask=False):
    print(labels_list, '\n', mask_id)

    if custom_mask:
        print('\n--------------------------------------CUSTOM MASKING STARTS--------------------------------------\n')
        unique_labels_list = labels_list.copy()
        unique_labels_list.remove('O')
        input_ids, entity_mask_ids, non_entity_mask_ids  = [], [], []
        for i, j in enumerate(tokenized_input):
            seq_ids, entity_masked_indices, masked_seq_ids, = j['input_ids'], [], []
            tokens = j['tokens']
            token_labels = [labels_list[k] for k in j['ner_tags']]
            initial_seq_ids_lens = len(seq_ids)
            assert len(tokens) == len(token_labels), "How comes length of tokens doesn't match langth of ner tags"
            #identifying_entity_using_label searches and finds entities from entire input document rather than a truncated version of the document
            seq_entity_labels = identify_entity_using_label(tokens, token_labels)
            token_ids = tokenizer.encode(" ".join(tokens))
            tokens_ = tokenizer.convert_ids_to_tokens(seq_ids)
            seq_entities = [i[0] for i in seq_entity_labels]
            discovered_entities = []
            if len(seq_entities) >= 1:
                try:
                    for q,entity in enumerate(seq_entities):
                        if entity.lower() not in discovered_entities:
                            entity_tok_ids = [tok for tok in tokenizer.encode(entity) if tok not in tokenizer.all_special_ids]
                            seq_ids, masked_indices = replace_id_with_mask_id(entity_tok_ids, mask_id, seq_ids, expand=True)
                            entity_masked_indices.append(masked_indices)
                            discovered_entities.append(entity.lower())
                            if not masked_indices and entity not in discovered_entities:
                                print("==============", q, entity, entity_tok_ids, masked_indices)
                except:
                    raise ValueError('Entity exists but not identified')
            elif len(seq_entities) == 0 and any(k in unique_labels_list for k in token_labels):
                print(tokens, '\n', token_labels, '\n', seq_entities)
                raise ValueError('Entity exists but not identified')

            seq_ids, non_entity_masked_indices = replace_random_id_with_mask(seq_ids, indices_masked=entity_masked_indices, tokenizer=tokenizer)
            final_seq_ids_lens = len(seq_ids)
            assert initial_seq_ids_lens == final_seq_ids_lens == len(j['ner_labels']) \
                   == len(j['attention_mask']) == len(j['labels']), \
                   "\n-----------------SOMETHING IS WRONG, CHECK OUT THE LENGTH OF THE TENSORS---------------\n"
            input_ids.append(seq_ids)
            entity_mask_ids.append(entity_masked_indices)
            non_entity_mask_ids.append(non_entity_masked_indices)

        tokenized_input_ = tokenized_input.remove_columns('input_ids')
        tokenized_input_ = tokenized_input_.add_column(name='input_ids', column=input_ids)
        tokenized_input_ = tokenized_input_.add_column(name='entity_specific_mask_ids', column=entity_mask_ids)
        tokenized_input_ = tokenized_input_.add_column(name='non_entity_specific_mask_ids', column=non_entity_mask_ids)
        n = 0

        print(type(tokenized_input_))
        for i, j in zip(tokenized_input, tokenized_input_):
            if n < 1:
                print([tokenizer.convert_ids_to_tokens(ids=[i for i in j['input_ids'] if i != tokenizer.pad_token_id])], len(j['input_ids']))
            n += 1
        return tokenized_input_
    else:
        for n,j in enumerate(tokenized_input):
            if n < 1:
                print([tokenizer.convert_ids_to_tokens(ids=[i for i in j['input_ids'] if i != tokenizer.pad_token_id])], len(j['input_ids']))
            n += 1
        print(type(tokenized_input))
        return tokenized_input

'''
    BASE LEVEL MASKING
    replace a random sequence of id's with a mask id or a replacement value within id's of an input sequence 
'''
def replace_random_id_with_mask(input_ids, indices_masked, tokenizer, mlm_prob=0.15):
    indices_of_masks = [ent[0] for msk_ents in indices_masked for ent in msk_ents]
    ids_masked = [id for msk_grp in indices_of_masks for id in msk_grp]
    non_entity_ids = [(i,j) for i,j in enumerate(input_ids) if j not in tokenizer.all_special_ids]
    rand = torch.rand(len(non_entity_ids)) < mlm_prob
    rand_ = torch.where(rand == 1)[0].tolist()

    ids_to_mask = [non_entity_ids[k] for k in rand_]
    for x in ids_to_mask:
        if x[0] not in ids_masked:
            input_ids[x[0]] = tokenizer.mask_token_id
        else:
            raise ValueError('Trying to mask an already masked id')

    return input_ids, ids_to_mask

'''
    ENTITY LEVEL MASKING
    replace a sequence of id's representing an entity with a mask id or a replacement value within id's of an input sequence 
'''
def replace_id_with_mask_id(sequence, replacement, lst, expand=False):
    new_list = lst.copy()
    masked_indices = []
    for i, e in enumerate(lst):
        if e == sequence[0]:
            # print("Sequence found", sequence)
            end = i
            f = 1
            for e1, e2 in zip(sequence, lst[i:]):
                if e1 != e2:
                    f = 0
                    break
                end += 1
            if f == 1:
                # del new_list[i:end]
                if expand:
                    # msk_ids = []
                    # for k in range(len(sequence)):
                    #     new_list.insert(i, replacement)
                    #     msk_ids.append(i+k)
                    # masked_indices.append((msk_ids, sequence))
                    assert len(range(i, end)) == len(sequence), "The sub array you're trying to replace is not of equal length with the sequence to be inserted"
                    new_list[i:end] = [replacement] * len(sequence)
                    msk_ids = [i+k for k in range(len(sequence))]
                    masked_indices.append((msk_ids, sequence))
                else:
                    # new_list.insert(i, replacement)
                    new_list[i] = replacement
                    masked_indices.append(i)
    return new_list, masked_indices

# extract an outcome based on the label
def identify_entity_using_label(tokens, token_labels):
    sentence_entities = []
    if len(tokens) == len(token_labels):
        e = 0
        for m in range(len(tokens)):
            if m == e:
                entity, entity_label = '', ''
                if token_labels[m].strip().startswith('B-'):
                    entity += tokens[m].strip()
                    entity_label += token_labels[m].strip()
                    e += 1
                    for w in range(m + 1, len(tokens)):
                        if token_labels[w].startswith('I-'):
                            entity += ' ' + tokens[w].strip()
                            entity_label += ' ' + token_labels[w].strip()
                            e += 1
                        else:
                            break
                    # print(entity, entity_label)
                    sentence_entities.append((entity, entity_label))
                else:
                    e += 1
    return sentence_entities

# check whether a sequence of ids is a subset of a larger sequence of ids
def isSubArray(A, B, n, m):
    # Two pointers to traverse the arrays
    i = 0
    j = 0
    # Traverse both arrays simultaneously
    while (i < n and j < m):
        # If element matches
        # increment both pointers
        if (A[i] == B[j]):
            i += 1
            j += 1
            # If array B is completely
            # traversed
            if (j == m):
                return True
        # If not,
        # increment i and reset j
        else:
            i = i - j + 1
            j = 0
    return False