# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 05/07/2021 
# @Contact: michealabaho265@gmail.com
import random
import re
from datasets import Dataset
import torch
from typing import List
import copy
from mslm import utils
import math
import random
import nltk
from nltk import *
from nltk.collocations import *
'''
    Masking specific tokens of the input dataset
'''

def customMask(tokenized_input, tokenizer, labels_mapping, custom_mask, random_mask, mlm_prob, elm_prob, strategy=None):
    print("Label mapping: ", labels_mapping, '\nMask token Id: ', tokenizer.mask_token_id)

    if custom_mask:
        print('\n--------------------------------------CUSTOM MASKING STARTS--------------------------------------\n')
        masked_input_ids, entity_mask_ids, non_entity_mask_ids  = [], [], []
        total_number_of_tokenized_ids = 0
        for i, j in enumerate(tokenized_input):
            seq_ids, entity_masked_indices, masked_seq_ids, = j['input_ids'], [], []
            total_number_of_tokenized_ids += len(seq_ids)
            initial_seq_ids_lens = len(seq_ids)
            assert len(seq_ids) == len(j['labels']), "How comes length of tokens doesn't match langth of ner labels"
            token_labels = [labels_mapping[n] if n in range(len(labels_mapping)) else str(n) for n in j['labels']]
            try:
                if strategy:
                    if strategy.lower() == 'span':
                        output_masked_input, non_entity_masked_indices = span_masking(input_ids=seq_ids,
                                                                                      tokenizer=tokenizer,
                                                                                      masking_rate=mlm_prob)
                else:
                    # identifying_entity_using_label searches, finds entities from entire input document and then replaces their token ids with a mask id
                    seq_ids, output_masked_input, seq_entity_labels = identify_entity_using_label(token_ids=seq_ids,
                                                                                                  token_labels=token_labels,
                                                                                                  tokenizer=tokenizer,
                                                                                                  elm_prob=elm_prob)

                    discovered_entities = []
                    if len(seq_entity_labels) >= 1:
                        for q,entity_extraction in enumerate(seq_entity_labels):
                            entity_toks, masked_indices, entity_tok_ids, entiy_tok_labels = entity_extraction
                            entity = tokenizer.convert_tokens_to_string(entity_toks)
                            entity_masked_indices.append(masked_indices)
                            if not masked_indices and entity.lower() not in discovered_entities:
                                print("==============", q, entity, entity_tok_ids, masked_indices)
                            discovered_entities.append(entity.lower())

                # replace_random_id with mask replaces an arbitrary set of tokens with a mask id
                if random_mask:
                    output_masked_input, non_entity_masked_indices = replace_random_id_with_mask(output_masked_input,
                                                                                                 indices_masked=entity_masked_indices,
                                                                                                 tokenizer=tokenizer,
                                                                                                 mlm_prob=mlm_prob)
            except:
                raise ValueError('Entity exists but not identified')

            final_seq_ids_lens = len(output_masked_input)

            assert initial_seq_ids_lens == final_seq_ids_lens == len(j['labels']) == len(j['attention_mask']) , \
                   "\n-----------------SOMETHING IS WRONG, CHECK OUT THE LENGTH OF THE TENSORS---------------\n"

            # print(len(output_masked_input), len(entity_masked_indices), len(non_entity_masked_indices))
            if output_masked_input:
                masked_input_ids.append(output_masked_input)
                indices_of_entity_masks = [ind for msk_inds in entity_masked_indices for ind in msk_inds]
                entity_masked_indices_ = [i if i in indices_of_entity_masks else 0 for i in range(len(seq_ids))]
                non_entity_masked_indices = [m[0] for m in non_entity_masked_indices]
                non_entity_masked_indices_ = [i if i in non_entity_masked_indices else 0 for i in range(len(seq_ids))]
                # print(len(output_masked_input), len(entity_masked_indices_), len(non_entity_masked_indices_))
                entity_mask_ids.append(entity_masked_indices_)
                non_entity_mask_ids.append(non_entity_masked_indices_)

        # tokenized_input_masked = tokenized_input.remove_columns('input_ids')
        tokenized_input = tokenized_input.add_column(name='masked_input_ids', column=masked_input_ids)
        tokenized_input = tokenized_input.add_column(name='entity_specific_mask_ids', column=entity_mask_ids)
        tokenized_input = tokenized_input.add_column(name='non_entity_specific_mask_ids', column=non_entity_mask_ids)

        for n,j in enumerate(tokenized_input):
            if n < 1:
                print([tokenizer.convert_ids_to_tokens(ids=[i for i in j['input_ids'] if i != tokenizer.pad_token_id])], len(j['input_ids']))
                print("\n")
            n += 1
        return tokenized_input
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
def replace_random_id_with_mask(input_ids, indices_masked, tokenizer, mlm_prob):
    indices_of_masks = [ind for msk_inds in indices_masked for ind in msk_inds]
    # ids_masked = [id for msk_grp in indices_of_masks for id in msk_grp]
    non_entity_ids = [(i,j) for i,j in enumerate(input_ids) if j not in tokenizer.all_special_ids]
    rand = torch.rand(len(non_entity_ids)) < mlm_prob
    rand_ = torch.where(rand == 1)[0].tolist()

    ids_to_mask = [non_entity_ids[k] for k in rand_]
    for x in ids_to_mask:
        if x[0] not in indices_of_masks:
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
            end = i
            f = 1
            for e1, e2 in zip(sequence, lst[i:]):
                if e1 != e2:
                    f = 0
                    break
                end += 1
            if f == 1:
                if expand:
                    assert len(range(i, end)) == len(sequence), "The sub array you're trying to replace is not of equal length with the sequence to be inserted"
                    new_list[i:end] = [replacement] * len(sequence)
                    msk_ids = [i+k for k in range(len(sequence))]
                    masked_indices.append((msk_ids, sequence))
                else:
                    new_list[i] = replacement
                    masked_indices.append(i)
    return new_list, masked_indices

# extract an outcome based on the label
def identify_entity_using_label(token_ids, token_labels, tokenizer, elm_prob):
    sentence_entities = []
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    token_ids_copy = copy.deepcopy(token_ids)

    #number of entities
    entities = [(i,j) for i,j in enumerate(token_labels) if j.startswith('B')]
    num_entities = len(entities)
    indices_of_entities_to_mask = []
    if num_entities > 0:
        num_entities_mask = elm_prob*num_entities #proportion of entities to mask
        if (num_entities_mask - int(num_entities)) >= 0.5:
            num_entities_mask = math.ceil(num_entities)
        else:
            num_entities_mask = int(num_entities_mask)
        entities_to_mask = random.sample(entities, num_entities_mask)
        indices_of_entities_to_mask.extend([i[0] for i in entities_to_mask])

    if len(tokens) == len(token_labels):
        e = 0
        for m in range(len(tokens)):
            if m == e:
                entity, entity_ids, entity_label, masked_indices = [], [], [], []
                if token_labels[m].strip().startswith('B') and m in indices_of_entities_to_mask:
                    entity.append(tokens[m].strip())
                    entity_ids.append(token_ids[m])
                    entity_label.append(token_labels[m].strip())
                    token_ids[m] = tokenizer.mask_token_id
                    masked_indices.append(m)
                    e += 1
                    for w in range(m + 1, len(tokens)):
                        if token_labels[w].startswith('I'):
                            entity.append(tokens[w].strip())
                            entity_ids.append(token_ids[w])
                            entity_label.append(token_labels[w].strip())
                            token_ids[w] = tokenizer.mask_token_id
                            masked_indices.append(w)
                            e += 1
                        else:
                            break
                    if not utils.isSubArray(token_ids_copy, entity_ids, len(token_ids_copy), len(entity_ids)):
                        raise ValueError("Ids of the identified entities are not within the tensor of input ids")
                    sentence_entities.append((entity, masked_indices, entity_ids, entity_label))
                else:
                    e += 1
        assert(len(indices_of_entities_to_mask) == len([i[0] for i in sentence_entities]))
    else:
        raise ValueError("Mismatch in sizes of the token tensor and token_labels tensor")
    return token_ids_copy, token_ids, sentence_entities

#random span masking
def span_masking(input_ids, tokenizer, masking_rate, max_length=5):
    non_masked_ids = [(i,j) for i,j in enumerate(input_ids) if j not in tokenizer.all_special_ids]
    original_input_len = len(non_masked_ids)+2
    n_mask_ids_locs, n_mask_ids = zip(*non_masked_ids)
    start_span_indices_pool = random.sample(range(len(input_ids)), len(input_ids))
    masking_budget = math.ceil(masking_rate * len(non_masked_ids))
    masked_so_far = 0
    span_ids_masked = []
    for x in start_span_indices_pool:
        #ensure its's not a special id
        if input_ids[x] in n_mask_ids:
            if not tokenizer.convert_ids_to_tokens(input_ids[x]).startswith("##"):
                #get a random length of span to mask
                random_span_len = torch.randint(1, max_length+1, (1,)).item()
                random_span_len = min(random_span_len, masking_budget)
                start, end = x, x+random_span_len
                # print("++", 'len:',random_span_len, "start:", start, "end:", end)
                if (masked_so_far + random_span_len) > masking_budget:
                    random_span_len = masking_budget - masked_so_far
                    end = x+random_span_len
                    # print("***exceed m_budget, new start and end", "start:", start, "end:", end)
                if end >= original_input_len-1:
                    end = original_input_len-1
                    random_span_len = end - start
                    # print("---end beyong sequence, new start and end", "start:", start, "end:", end)
                if not any(i==tokenizer.mask_token_id for i in input_ids[start:end]):
                    input_ids[start:end] = [tokenizer.mask_token_id] * random_span_len
                    span_ids_masked += list(zip(range(start, end), input_ids[start:end]))
                    random_span_len = end - start
                    # print('len:',random_span_len, "start:", start, "end:", end)
                    masked_so_far += random_span_len
                    # print(input_ids)
        else:
            pass
        if masked_so_far >= masking_budget:
            break
    return input_ids, span_ids_masked

