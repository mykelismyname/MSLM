import os.path

import nltk
from nltk import *
from nltk.collocations import *
from mslm import utils
import heapq

#pmi-masking, masking tokens with a high pointwise mutual information score
def construct_pmi_masking_vocabulary(processed_data, tokenizer, dataset_name):
    corpus_tokens = []
    for i in processed_data:
        i_tokens = [tok for seq in processed_data[i]['tokens'] for tok in seq]
        corpus_tokens.extend(i_tokens)

    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend(tokenizer.all_special_tokens)

    ngram_segmenations = {"bigrams":[], "trigrams":[], "quadgrams":[]}

    bigram_measures = BigramAssocMeasures()
    trigram_measures = TrigramAssocMeasures()
    quadgram_measures = QuadgramAssocMeasures()

    bifinder = BigramCollocationFinder.from_words(corpus_tokens)
    bifinder.apply_word_filter(lambda w: len(w)<2 or w.lower() in stop_words)
    bifinder.apply_freq_filter(min_freq=5)
    bi_pmi_scores = bifinder.score_ngrams(bigram_measures.pmi)

    trifinder = TrigramCollocationFinder.from_words(corpus_tokens)
    trifinder.apply_word_filter(lambda w: len(w) < 2 or w.lower() in stop_words)
    trifinder.apply_freq_filter(min_freq=3)
    tri_pmi_scores = trifinder.score_ngrams(trigram_measures.pmi)

    quadfinder = QuadgramCollocationFinder.from_words(corpus_tokens)
    quadfinder.apply_word_filter(lambda w: len(w) < 2 or w.lower() in stop_words)
    quadfinder.apply_freq_filter(min_freq=2)
    quad_pmi_scores = quadfinder.score_ngrams(quadgram_measures.pmi)

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

    output_dir = "mslm/masking/pmi_masking_vocabularly"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.basename(os.path.dirname(dataset_name))

    with open(output_dir+"/{}.txt".format(output_file), 'w') as writer:
        for gram in ngram_segmenations:
            for ent in ngram_segmenations[gram]:
                gram_ent = " ".join(list(ent[0]))
                writer.write("{}\n".format(gram_ent))
        writer.close()
