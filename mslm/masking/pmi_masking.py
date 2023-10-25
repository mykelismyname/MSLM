import os.path

import nltk
import os
from nltk import *
from nltk.collocations import *
import sys
sys.path.append("../")
from mslm import utils
import heapq
from transformers import AutoTokenizer
from datasets import load_dataset
from glob import glob
from argparse import ArgumentParser

#pmi-masking, masking tokens with a high pointwise mutual information score
class PMI_Vocab:
    def __init__(self,
                 data_path,
                 model,
                 dataset_name):
        """
        Args:
            data_path: Path to the dataset whose pmi vocabilarly to create
            model: pretrained tokenizer model
            dataset_name: name of the dataset to use in outputting vovab file
        """
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.train = "train.txt"
        self.dev = "devel.txt"
        self.test = "test.txt"

    def load_data(self, use_hugging_face=False):
        """
        Args:
            use_hugging_face: use huggingface dataset loader to load the raw data from path
        Returns: a list of tokens sequentially ordered from first to last sentence in the data
        """
        if use_hugging_face:
            dataset_path_files = os.listdir(os.path.abspath(self.data_path))
            data_files = {}
            if self.train in dataset_path_files:
                data_files["train"] = os.path.abspath(self.data_path+"/"+self.train)
            if self.dev in dataset_path_files:
                data_files["validation"] = os.path.abspath(self.data_path+"/"+self.dev)
            if self.test in dataset_path_files:
                data_files["test"] = os.path.abspath(self.data_path+"/"+self.test)
            print(data_files)
            raw_datasets = load_dataset(args.loading_dataset_script, data_files=data_files)
            corpus_tokens = []
            for i in raw_datasets:
                i_tokens = [tok for seq in raw_datasets[i]['tokens'] for tok in seq]
                corpus_tokens.extend(i_tokens)
        else:
            pass
        return corpus_tokens

    def construct_vocabularly(self, min_freq, use_hugging_face=False):
        corpus = self.load_data(use_hugging_face=use_hugging_face)
        print(self.dataset_name)
        construct_pmi_masking_vocabulary(corpus=corpus,
                                         tokenizer=self.tokenizer,
                                         dataset_name=self.dataset_name,
                                         min_freq=min_freq)

def CollogramFinder(Finder, tokenizer, AssocMeasure, corpus_tokens, min_freq):
    """
    Args:
        Finder:
        AssocMeasure: PMI association measure from nltk https://www.nltk.org/_modules/nltk/metrics/association.html
        corpus_tokens: list of tokens of dataset sentences ordered from first sentence token to last sentence token
        min_freq: minimum occurrence frequency of an ngram collocation
    Returns: ngram collocation with their corresponding scores
    """
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend(tokenizer.all_special_tokens)

    _finder_ = Finder.from_words(corpus_tokens)
    _finder_.apply_word_filter(lambda w: len(w) < 2 or w.lower() in stop_words)
    _finder_.apply_freq_filter(min_freq=min_freq)
    pmi_scores = _finder_.score_ngrams(AssocMeasure.pmi)
    return pmi_scores

def construct_pmi_masking_vocabulary(corpus, tokenizer, dataset_name, min_freq):
    """
    Args:
        corpus: list of tokens of dataset sentences ordered from first sentence token to last sentence token
        tokenizer:
        dataset_name:
    Returns: pmi vocabularly file
    """
    ngram_segmenations = {"bigrams":[], "trigrams":[], "quadgrams":[]}

    bigram_measures = BigramAssocMeasures()
    bi_pmi_scores = CollogramFinder(Finder=BigramCollocationFinder, tokenizer=tokenizer, AssocMeasure=bigram_measures, corpus_tokens=corpus, min_freq=min_freq[0])
    trigram_measures = TrigramAssocMeasures()
    tri_pmi_scores = CollogramFinder(Finder=TrigramCollocationFinder, tokenizer=tokenizer, AssocMeasure=trigram_measures, corpus_tokens=corpus, min_freq=min_freq[1])
    quadgram_measures = QuadgramAssocMeasures()
    quad_pmi_scores = CollogramFinder(Finder=QuadgramCollocationFinder, tokenizer=tokenizer, AssocMeasure=quadgram_measures, corpus_tokens=corpus, min_freq=min_freq[2])

    max_ngram_length = max(len(bi_pmi_scores), len(tri_pmi_scores), len(quad_pmi_scores))

    for n in range(max_ngram_length):
        try:
            ngram_segmenations["bigrams"].append(bi_pmi_scores[n])
            ngram_segmenations["trigrams"].append(tri_pmi_scores[n])
            ngram_segmenations["quadgrams"].append(quad_pmi_scores[n])
        except Exception as e:
            pass

    collocations_lists = {"bi": [], "tri":[], "quad":[]}
    for i, collocation in enumerate(ngram_segmenations["quadgrams"]):
        quad_colloc, quad_colloc_score = collocation
        quad_colloc = list(quad_colloc)
        colloc_entries = [collocation]
        #check trigrams for collocations weaker link
        for tri_collocation in ngram_segmenations["trigrams"]:
            tri_colloc, tri_colloc_score = tri_collocation
            tri_colloc = list(tri_colloc)
            if utils.isSubsequence(tri_colloc, quad_colloc):
                if tri_colloc_score < quad_colloc_score:
                    if tri_collocation not in colloc_entries:
                        colloc_entries.insert(0, tri_collocation)
                    # check bigrams for collocations weaker link
                    for bi_colocation in ngram_segmenations["bigrams"]:
                        bi_colloc, bi_colloc_score = bi_colocation
                        bi_colloc = list(bi_colloc)
                        if utils.isSubsequence(bi_colloc, tri_colloc):
                            if bi_colloc_score < tri_colloc_score:
                                if bi_colocation not in colloc_entries:
                                    colloc_entries.insert(0, bi_colocation)
                            else:
                                print("Tri collocation appears to have a smaller PMI score than bigram collocation")
                else:
                    print("Quad collocation appears to have a smaller PMI score than Trigram collocation")
        if i == 0:
            heapq.heapify(collocations_lists["bi"])
            heapq.heapify(collocations_lists["tri"])
            heapq.heapify(collocations_lists["quad"])

        for entry in colloc_entries:
            if len(entry) == 2 and entry not in collocations_lists["bi"]:
                heapq.heappush(collocations_lists["bi"], entry)
            if len(entry) == 3 and entry not in collocations_lists["tri"]:
                heapq.heappush(collocations_lists["tri"], entry)
            if len(entry) == 4 and entry not in collocations_lists["quad"]:
                heapq.heappush(collocations_lists["quad"], entry)

    output_dir = "../mslm/masking/pmi_masking_vocabularly"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # output_file = os.path.basename(os.path.dirname(dataset_name))

    with open(output_dir+"/{}.txt".format(dataset_name), 'w') as writer:
        for gram in ngram_segmenations:
            for ent in ngram_segmenations[gram]:
                gram_ent = " ".join(list(ent[0]))
                writer.write("{}\n".format(gram_ent))
        writer.close()

def main():
    return

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

if __name__ == "__main__":
    par = ArgumentParser()
    par.add_argument("--min_freq", type=list_of_ints, default=[5,3,2], help="minimum occurrence frequency of bigrams, trigrams and quadgrams")
    par.add_argument("--data_path", type=str, help="path to dataset")
    par.add_argument("--dataset_name", type=str, help="path to dataset")
    par.add_argument("--model", type=str, help="path to pretraned model")
    par.add_argument("--loading_dataset_script", type=str)
    args = par.parse_args()

    pmi = PMI_Vocab(data_path=args.data_path, model=args.model, dataset_name=args.dataset_name)
    pmi.construct_vocabularly(min_freq=args.min_freq, use_hugging_face=True)