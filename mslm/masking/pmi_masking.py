import nltk
from nltk import *
from nltk.collocations import *

#pmi-masking, masking tokens with a high pointwise mutual information score
def construct_pmi_masking_vocabulary(processed_data, tokenizer):
    corpus_tokens = []
    for i in processed_data:
        i_tokens = [tok for seq in processed_data[i]['tokens'] for tok in seq]
        corpus_tokens.extend(i_tokens)
    # corpus_tokens = [tok for seq in input_corpus['tokens'] for tok in seq]
    print(corpus_tokens[:80])

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
    trifinder.apply_freq_filter(min_freq=5)
    tri_pmi_scores = trifinder.score_ngrams(trigram_measures.pmi)

    quadfinder = QuadgramCollocationFinder.from_words(corpus_tokens)
    quadfinder.apply_word_filter(lambda w: len(w) < 2 or w.lower() in stop_words)
    quadfinder.apply_freq_filter(min_freq=5)
    quad_pmi_scores = quadfinder.score_ngrams(quadgram_measures.pmi)

    max_ngram_length = max(len(bi_pmi_scores), len(tri_pmi_scores), len(quad_pmi_scores))
    print(len(bi_pmi_scores), len(tri_pmi_scores), len(quad_pmi_scores), max_ngram_length)

    for n in range(max_ngram_length):
        try:
            ngram_segmenations["bigrams"].append(bi_pmi_scores[n])
            ngram_segmenations["trigrams"].append(tri_pmi_scores[n])
            ngram_segmenations["quadgrams"].append(quad_pmi_scores[n])
        except Exception as e:
            pass

    collocations_list = []
    for collocation in ngram_segmenations["quadgrams"]:
        quad_colloc, quad_colloc_score = collocation
        quad_colloc = list(quad_colloc)
        colloc_entries = [collocation]
        #check trigrams for collocations weaker link
        for tri_collocation in ngram_segmenations["trigrams"]:
            tri_colloc, tri_colloc_score = tri_collocation
            tri_colloc = list(tri_colloc)
            if util.isSubsequence(tri_colloc, quad_colloc):
                if tri_colloc_score < quad_colloc_score:
                    colloc_entries.insert(0, tri_collocation)
                    for bi_colocation in ngram_segmenations["bigrams"]:
                        bi_colloc, bi_colloc_score = bi_colocation
                        bi_colloc = list(bi_colloc)
                        if util.isSubsequence(bi_colloc, tri_colloc):
                            if bi_colloc < tri_colloc:
                                colloc_entries.insert(0, bi_colocation)
                            else:
                                print("Tri collocation appears to have a smaller PMI score than bigram collocation")
                else:
                    print("Quad collocation appears to have a smaller PMI score than Trigram collocation")
        collocations_list.append(colloc_entries)
    
    for i,j in ngram_segmenations.items():
        print(i)
        print(j[:10])
        print("..........................................................")