import os
import pickle
import re
import sys
import numpy as np
import datasets
import json
import pandas as pd
import torch
import sklearn.metrics as sk
from copy import deepcopy
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

_LABELS = {"Disease": ["Acquired Abnormality", "Anatomical Abnormality", "Bacterium", "Archaeon", "Congenital Abnormality",
           "Cell or Molecular Dysfunction", "Disease or Syndrome", "Virus", "Mental or Behavioral Dysfunction",
           "Pathologic Function", "Injury or Poisoning"],
           "Symptom":["Social Behavior", "Sign or Symptom"],
           "Treatment":["Pharmacologic Substance", "Clinical Drug"]}

#create a dataset with BIO tags for each entity
def create_ner_datasets(data_dir, dest_dir):
    index_ids = np.random.permutation(12422)
    train_len = int(0.8 * len(index_ids))
    dev_len = int(0.1 * len(index_ids))
    train_ids = index_ids[:train_len]
    dev_ids = index_ids[train_len:train_len+dev_len]
    idx_counter = 0
    dest_dir = os.path.abspath(dest_dir)
    label_list = [v for k,v in _LABELS.items()]
    label_list = [i for j in label_list for i in j]
    with open(dest_dir+"/train.txt", "w") as tr, open(dest_dir+"/dev.txt", "w") as de, open(dest_dir+"/test.txt", "w") as te:
        for data_file in os.listdir(data_dir):
            data_file = os.path.join(os.path.abspath(data_dir), data_file)
            if os.path.basename(data_file).endswith('.pkl'):
                data = pickle.load(open(data_file, "rb"))
            if os.path.basename(data_file).endswith('.json'):
                data = json.load(open(data_file, "r"))
            if os.path.basename(data_file).endswith('.txt'):
                data = open(data_file, "r").readlines()

            if os.path.basename(data_file).endswith('.pkl') or os.path.basename(data_file).endswith('.json'):
                print(data_file)
                for idx, doc in enumerate(data):
                    if doc:
                        doc_tokens = [tok for d in doc['Sents'] for tok in d]
                        doc_ents_tags = ["O" for _ in range(len(doc_tokens))]
                        for ent in doc['Entities']:
                            # linked_entities = tuple(sorted(ent['linked_umls_entities'].items(), key=lambda x:x[1]['score'], reverse=True))
                            linked_entities = tuple(ent['linked_umls_entities'].items())
                            try:
                                tag = linked_entities[0][1]['type']
                                if tag in label_list:
                                    start, end = ent['pos'][0], ent['pos'][1]
                                    if tag in _LABELS["Disease"]:
                                        _tag_ = "Disease"
                                    elif tag in _LABELS["Symptom"]:
                                        _tag_ = "Symptom"
                                    elif tag in _LABELS["Treatment"]:
                                        _tag_ = "Treatment"
                                    else:
                                        print()
                                        raise ValueError(f"label {tag} in wrong place")
                                    ent_tags = [f"I-{_tag_}" for _ in range(start, end)]
                                    ent_tags[0] = f"B-{_tag_}"
                                    doc_ents_tags[start:end] = ent_tags
                            except IndexError:
                                pass

                        assert len(doc_tokens) == len(doc_ents_tags)
                        if idx_counter in train_ids:
                            for x,y in zip(doc_tokens, doc_ents_tags):
                                tr.write("{} {}\n".format(x, y))
                            tr.write("\n")
                        elif idx_counter in dev_ids:
                            for x, y in zip(doc_tokens, doc_ents_tags):
                                de.write("{} {}\n".format(x, y))
                            de.write("\n")
                        else:
                            for x, y in zip(doc_tokens, doc_ents_tags):
                                te.write("{} {}\n".format(x, y))
                            te.write("\n")
                        idx_counter += 1
                    else:
                        print("here")
                    if idx == len(data)-1:
                        print(f"Final Doc-{idx} and idx counter is {idx_counter}")


#create a file with a list of all B,I,O labels within the  dataset
def create_ner_labels(file, use_dataset=False):
    labels_file = os.path.join(os.path.dirname(file), "labels.json")
    labels = []
    if use_dataset:
        file_path = os.path.abspath(file)
        if os.path.isdir(file_path):
            files = [os.path.join(file_path, i) for i in os.listdir(file_path)]
        else:
            files = [file_path]
        print(files)
        for file_ in files:
            extension = os.path.basename(file_).split(".")[-1]
            if extension == 'pkl':
                data = pickle.load(open(os.path.abspath(file_), "rb"))
            if extension == 'json':
                data = json.load(open(os.path.abspath(file_), "r"))
            for idx, doc in enumerate(data):
                if doc:
                    for e, ent in enumerate(doc['Entities']):
                        linked_entities = tuple(ent['linked_umls_entities'].items())
                        try:
                            label = linked_entities[0][1]['type']
                            b_label = "B-" + label
                            if b_label not in labels:
                                labels.append(b_label)
                            i_label = "I-" + label
                            if i_label not in labels:
                                labels.append(i_label)
                        except IndexError:
                            pass
    else:
        with open(file, 'r') as f:
            for line in f.readlines():
                line_split = line.split("|", 2)
                label = line_split[-1].strip()
                labels.append("B-"+label)
                labels.append("I-"+label)

    with open(labels_file, "w") as d:
        labels = sorted(list(set(labels)))
        labels.insert(0, "O")
        labels_dict = {"labels": labels}
        json.dump(labels_dict, d, indent=2)
        d.close()

def read_data(filepath):
    if filepath.endswith('.pkl'):
        data = pickle.load(open(os.path.abspath(filepath), "rb"))
    if filepath.endswith('.json'):
        data = json.load(open(os.path.abspath(filepath), "r"))
    return data

#create a structured dataset with entities classified under different semantic typoes
def create_structured_data(data_dir, labels_file):
    if os.path.isdir(data_dir):
        data_files = [read_data(os.path.join(os.path.abspath(data_dir),i)) for i in os.listdir(data_dir)]
    else:
        data_files = [read_data(data_dir)]
    labels = [i for i in json.load(open(labels_file, 'r'))['labels'] if i.startswith('B-')]
    labels = [i.split('B-')[-1] for i in labels]
    print(labels, "\n")
    dataset = []
    # with open(dest_file, "w") as d:
    patient_id = 1
    for f, file in enumerate(data_files):
        for idx, doc in enumerate(file):
            if doc:
                doct_types_entities_codes_dict = {'patient_id':[int(patient_id)]}
                for ent in doc['Entities']:
                    linked_entities = tuple(ent['linked_umls_entities'].items())
                    try:
                        linked_cat = linked_entities[0] #selecting entity with highest confidence score
                        type = linked_cat[1]['type']
                        snomed_code = linked_cat[0]
                        entity = ent['name'].lower()
                        if type not in doct_types_entities_codes_dict:
                            doct_types_entities_codes_dict[type] = [entity]
                            doct_types_entities_codes_dict[type+'_snomed_code'] = [snomed_code]
                        else:
                            if entity not in doct_types_entities_codes_dict[type]:
                                doct_types_entities_codes_dict[type].append(entity)
                                doct_types_entities_codes_dict[type+'_snomed_code'].append(snomed_code)
                    except IndexError:
                        pass

                doct_types_entities_codes_dict_df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in doct_types_entities_codes_dict.items()]))
                print(doct_types_entities_codes_dict_df)
                dataset.append(doct_types_entities_codes_dict_df)
                patient_id += 1
                print("\n")

    patient_dataset_df = pd.concat(dataset)
    output_dir = os.path.abspath('../data/structured_data/')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    patient_dataset_df.to_csv(os.path.join(output_dir, 'patient_data.csv'))

# check whether a sequence of ids is a subset of a larger sequence of ids
def isSubArray(A, B, n, m):
    i = 0
    j = 0
    while (i < n and j < m):
        if (A[i] == B[j]):
            i += 1
            j += 1
            if (j == m):
                return True
        else:
            i = i - j + 1
            j = 0
    return False

#check if span or sequence is a sub-span or subsequence of a large span or sequence respectively
def isSubsequence(s: str, t: str) -> bool:
    i, j = 0, 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
        j += 1
    return i == len(s)

#extra padding to use inrder to have an equal number of rows within tokenized input
def padding(input, max_length, pad_token):
    pad_tokens = [pad_token]*(max_length - len(input))
    input += pad_tokens
    return input

#send tensor to available device
def device(inp):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return inp.to(device)

#create batch with only items necessary for the mask language models to encode
def batch_input_creation(batch_input):
    batch = {k: v for k, v in batch_input.items() if k not in ['entity_specific_mask_ids',
                                                                'non_entity_specific_mask_ids',
                                                                'input_ids',
                                                                'masked_input_ids']}
    batch['input_ids'] = batch_input.pop('masked_input_ids')
    return batch

#compuyting an exact_match score for named entity recognition
def exact_match_ner(refs, preds, id2label, input_ids, tokenizer, score = 0):
    input_ids = input_ids.cpu().tolist()
    refs = refs.cpu().tolist()
    preds = preds.cpu().tolist()
    assert len(refs) == len(preds) == len(input_ids)
    num_samples = len(input_ids)

    tokens = [tokenizer.convert_ids_to_tokens(i) for i in input_ids]
    predictions = []
    match_entity_count, total_entity_count = 0, 0
    for x in range(num_samples):
        toks = [t for t in tokens[x] if t != '[PAD]']
        inds_to_remove = [ind for ind,l in enumerate(refs[x][:len(toks)]) if l == -100]
        _refs_ = [l for k,l in enumerate(refs[x][:len(toks)]) if k not in inds_to_remove]
        _preds_ = [l for k,l in enumerate(preds[x][:len(toks)]) if k not in inds_to_remove]
        _toks_ = [l for k,l in enumerate(toks) if k not in inds_to_remove]
        seq_length = len(_refs_)
        n = 0
        for y in range(seq_length):
            predictions.append("{} {} {}".format(_toks_[y], id2label[_preds_[y]], id2label[_refs_[y]]))
            if y == n:
                if id2label[_refs_[y]].startswith("B"):
                    entity = [_toks_[y]]
                    # print('++++',_refs_[y], _preds_[y], _toks_[y])
                    for z in range(y+1, seq_length):
                        if id2label[_refs_[z]].startswith("B") and _toks_[z].startswith("##"):
                            n += 1
                            entity.append(_toks_[z])
                        elif id2label[_refs_[z]].startswith("I"):
                            n += 1
                            entity.append(_toks_[z])
                        else:
                            break
                    n += 1
                    # print(_refs_[y:n], _preds_[y:n], y, n, entity)
                    if _refs_[y:n] == _preds_[y:n]:
                        # print("=======",_refs_[y:n], _preds_[y:n], _toks_[y:n], entity)
                        match_entity_count += 1
                    total_entity_count += 1
                else:
                    n += 1
        predictions.append("\n")
    if match_entity_count > 0:
        score = float(match_entity_count/total_entity_count)
    return predictions, (total_entity_count, match_entity_count, score)

#retrieve tensors with ids of entity-level, base-level masks as well as non-masked tokens
def identify_tokens_with_and_without_masks(datasets):
    datasets_masked_ids = []
    for dataset in datasets:
        entity_mask = []
        non_entity_mask = []
        non_masked = []
        for d in dataset:
            batch_size = d['input_ids'].size(0)
            d_entity_mask, d_non_entity_mask, d_non_masked = [], [], []
            for k in range(batch_size):
                entity_mask_no = d['entity_specific_mask_ids'][k]
                non_entity_mask_no = d['non_entity_specific_mask_ids'][k]
                entity_mask_no = entity_mask_no > 0
                entity_mask_no_list = torch.flatten(entity_mask_no.nonzero()).tolist()
                non_entity_mask_no = non_entity_mask_no > 0
                non_entity_mask_no_list = torch.flatten(non_entity_mask_no.nonzero()).tolist()
                non_mask_list = [i for i in range(len(entity_mask_no)) if
                                 i not in entity_mask_no_list + non_entity_mask_no_list]
                d_entity_mask.append(entity_mask_no_list)
                d_non_entity_mask.append(non_entity_mask_no_list)
                d_non_masked.append(non_mask_list)
            entity_mask.append(d_entity_mask)
            non_entity_mask.append(d_non_entity_mask)
            non_masked.append(d_non_masked)
        datasets_masked_ids.append([entity_mask, non_entity_mask, non_masked])
    return datasets_masked_ids

#calculate a weight for the arbitrary masked tokens (base level masking), entity masked tokens
# (entity-level masking) as well as non-masked tokens
def compute_weights(train_ids, eval_ids, include_non_mask_tokens=False):
    def flatten_batch_of_lists(x):
        output = []
        for batch in x:
            for seq in batch:
                for index in seq:
                    output.append(index)
        return output

    def compute_weight_matrix(list_of_mask_type_counts):
        list_of_mask_type_counts = list_of_mask_type_counts[:2]
        total_count = sum(list_of_mask_type_counts)
        weight_matrix = torch.zeros(len(list_of_mask_type_counts))
        for n, m in enumerate(list_of_mask_type_counts):
            # if n == 0:
            #     weight_matrix[n] = np.sqrt(m / total_count)
            # else:
            weight_matrix[n] = 1 - float(m / total_count)
        weight_matrix[0] = max(0.5, weight_matrix[1])
        weight_matrix[1] = min(0.5, weight_matrix[1])
        sft = torch.nn.Softmax(dim=0)
        weight_matrix = sft(weight_matrix)
        return weight_matrix

    len_e_ms, len_ne_ms, len_n_ms = 0, 0, 0 #number of entity tokens masked, arbitrary masked tokens and non masked tokens
    for d in [train_ids, eval_ids]:
        e, f, g = d
        len_e_ms += len(flatten_batch_of_lists(e))
        len_ne_ms += len(flatten_batch_of_lists(f))
        len_n_ms += len(flatten_batch_of_lists(g))

    print(f"Number of entity tokens masked: {len_e_ms}")
    print(f"Number of arbtrary masked tokens: {len_ne_ms}")
    print(f"Number of non masked tokens: {len_n_ms}")

    if include_non_mask_tokens:
        return compute_weight_matrix([len_e_ms, len_ne_ms, len_n_ms])

    return compute_weight_matrix([len_e_ms, len_ne_ms])

#compute a tensor with a weight for each token with respect to the mask (mask specific weight)
def compute_mask_specific_weights(data, batch_size, seq_len, weight_matrix):
    entity_masked_data, non_entity_masked_data, non_masked_data = data
    assert len(entity_masked_data) == len(non_entity_masked_data) == len(non_masked_data), "Something is wrong"
    n = len(entity_masked_data)
    weights = torch.zeros(n, batch_size, seq_len)
    for b in range(n):
        for i in range(len(entity_masked_data[b])):
            weights[b][i, entity_masked_data[b][i]] = weight_matrix[0]
            weights[b][i, non_entity_masked_data[b][i]] = weight_matrix[1]
            weights[b][i, non_masked_data[b][i]] = 1
    return weights

#compute lengths of sentences
def compute_sentence_length(data_dir):
    data_files = [i for i in glob(data_dir+"/*.txt") if os.path.basename(i) in ['train.txt', 'test.txt', 'dev.txt', 'devel.txt']]
    sent_lengths = []
    entity_lens = []
    dataset_entity_mentions = 0
    for data_file in data_files:
        print(data_file)
        max_length = 0
        data_file_entity_mentions = 0
        with open(data_file, 'r') as g:
            data = g.readlines()
            tokens, labels = [], []
            longest_sentence = None
            sentence_counter = 0
            for i,line in enumerate(data):
                if line == '\n':
                    length = len(tokens)
                    sent_lengths.append(length)
                    if length > max_length:
                        if length > 0:
                            max_length = length
                            longest_sentence = " ".join([k for k in tokens])
                    sentence_counter += 1
                    sent_entity_mentions = [j for j in labels if j.startswith('B')]

                    for m in range(len(labels)):
                        if labels[m].startswith('B'):
                            entity = [tokens[m]]
                            for w in range(m+1, len(labels)):
                                if labels[w].startswith('I'):
                                    entity.append(tokens[w])
                                else:
                                    break
                            if len(entity) > 0:
                                entity_lens.append(len(entity))

                    if len(sent_entity_mentions) > 0:
                        data_file_entity_mentions += len(sent_entity_mentions)
                    tokens.clear()
                    labels.clear()
                elif line != " ":
                    line = line.split()
                    tokens.append(line[0])
                    labels.append(line[1])
                else:
                    print("Line-", line)
            print(f"{data_file} has {sentence_counter} sentences and the  longest sentence is {max_length} words long"
                  f" and {data_file_entity_mentions} entity mentions")
            dataset_entity_mentions += data_file_entity_mentions
            g.close()
    print(f"Average sentence length {np.mean(sent_lengths)}")
    print(f"Min sentence length {np.min(sent_lengths)}")
    print(f"Max Sentence length {np.max(sent_lengths)}")
    print(f"The dataset has got {dataset_entity_mentions} entity mentions and average entity mention length {np.mean(entity_lens)}")

#plotting the f1, perplexity, loss over training time
def plot_metrics(result_dirs, metric):
    res_dirs = os.listdir(result_dirs)
    res_dirs_path = os.path.abspath(result_dirs)
    if metric in ["loss", "perplexity"]:
        legend_loc = "upper right"
    else:
        legend_loc = "lower right"
    for d in res_dirs:
        d_loc = os.path.join(res_dirs_path, d)
        try:
            x = json.load(open(d_loc+"/tracked_metrics.json", 'r'))
            f1_scores = x[metric]
            x_values = [i+1 for i in range(len(f1_scores))]
            plt.plot(x_values, f1_scores, label=d)
            plt.legend(loc=legend_loc)
            plt.xticks(x_values)
        except OSError as e:
            print("{} desn't have tracked metrics".format(d))
            pass
    plt.xlabel("epochs")
    plt.ylabel(metric)
    plt.savefig(os.path.join(res_dirs_path, "{}.png".format(metric)))
    plt.show()

def main():
    task = input("What do you want, create_ner_dataset or create_ner_labels or create_structured_dataset ?\n")
    if task == "create_ner_dataset":
        source = input("Enter the dataset path ?\n")
        dest = input("Enter the path to an output file ?\n")
        create_ner_datasets(source, dest)
    if task == "create_ner_labels":
        source = input("Create from a dataset or a file with semantic types ?\n")
        file = input("Enter path to file ?\n")
        if source == "dataset":
            create_ner_labels(file, use_dataset=True)
        else:
            create_ner_labels(file)
    if task == "create_structured_dataset":
        source = input("Enter the dataset dir or file path ?\n")
        labels = input("Enter the labels file path ?\n")
        create_structured_data(source, labels)
    if task == "compute_max_length":
        source = input("Enter the dataset dir or file path ?\n")
        compute_sentence_length(source)
if __name__ == "__main__":
    # main()
    import sys
    args = sys.argv
    print(args)
    compute_sentence_length(args[1])
    # plot_metrics(args[1], args[2])
    # create_ner_datasets(data_dir=args[1], dest_dir=args[2])
    # p = os.path.abspath(args[1])
    # labels = []
    # for t in os.listdir(p):
    #     print(t)
    #     if t in ['train.txt', 'test.txt', 'devel.txt']:
    #         with open(os.path.join(p, t), 'r') as s:
    #             for l in s.readlines():
    #                 if l != '\n':
    #                     l = l.split()
    #                     if l[1] not in labels:
    #                         labels.append(l[1].strip())
    # print(labels)
