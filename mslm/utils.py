import os
import pickle
import re
import sys
import numpy as np
import datasets
import json
import pandas as pd

#create a dataset with BIO tags for each entity
def create_ner_datasets(data_file, dest_file):
    with open(data_file, "r") as f, open(dest_file, "w") as d:
        data = json.load(f)
        for idx, doc in enumerate(data):
            if doc:
                doc_ents = []
                for s, sent in enumerate(doc['Sents']):
                    sent_ents_tags = ["O" for _ in range(len(sent))]
                    sent_ents = [ent for ent in doc['Entities'] if ent['sent_id'] == s]
                    for ent in sent_ents:
                        # linked_entities = tuple(sorted(ent['linked_umls_entities'].items(), key=lambda x:x[1]['score'], reverse=True))
                        linked_entities = tuple(ent['linked_umls_entities'].items())
                        try:
                            tag = linked_entities[0][1]['type']
                            start, end = ent['pos'][0], ent['pos'][1]
                            ent_tags = [f"I-{tag}" for i in range(start, end)]
                            ent_tags[0] = f"B-{tag}"
                            sent_ents_tags[start:end] = ent_tags
                        except IndexError:
                            pass
                    doc_ents.append(sent_ents_tags)
                    for x,y in zip(sent, sent_ents_tags):
                        d.write("{} {}\n".format(x,y))
                    d.write("\n")
                d.write("\n---doc_end---\n")

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

if __name__ == "__main__":
    main()