import os
import pickle
import re
import sys
import numpy as np
import datasets
import json

def create_ner_datasets(data_file):
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

def main():
    task = input("What do you want, create_ner_dataset or create_ner_labels ?\n")
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

if __name__ == "__main__":
    main()