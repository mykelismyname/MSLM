"""TODO(wikitext): Add a description here."""


import os
import pickle
import sys
import numpy as np
import datasets
import json

_DESCRIPTION = """\
 A biomedical corpus containing Electronic Health Records (EHRs).
"""
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""
_DATA_URL = ""
_LABELS = ["O", "B", "I"]

class EhrdataConfig(datasets.BuilderConfig):
    """BuilderConfig for GLUE."""

    def __init__(self, data_url, features, **kwargs):
        """BuilderConfig for Wikitext

        Args:
          data_url: `string`, url to the dataset (word or raw level)
          **kwargs: keyword arguments forwarded to super.
        """
        super(EhrdataConfig, self).__init__(
            version=datasets.Version(
                "1.0.0",
            ),
            **kwargs,
        )
        self.data_url = data_url
        self.features = features


class Ehrdatatext(datasets.GeneratorBasedBuilder):
    """Dataset of electronic health records (mimic-III) weakly annotated using scispacy"""

    VERSION = datasets.Version("0.1.0")
    BUILDER_CONFIGS = [
        EhrdataConfig(
            name="mimic-ii",
            data_url=_DATA_URL + "/" + "wikitext-103-v1.zip",
            features=["text", "tokens", "ner_tags"],
            description="Word level dataset. No processing is needed other than replacing newlines with <eos> tokens.",
        ),
    ]

    def _info(self):
        # TODO(wikitext): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=_LABELS
                        )
                    ),
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(wikitext): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        data_urls = self.config.data_files
        downloaded_files = dl_manager.download_and_extract(data_urls)
        print("Downloaded files:", downloaded_files)
        if 'test' in downloaded_files:
            return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                        gen_kwargs={"filepath": downloaded_files["train"][0]}),
                datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                        gen_kwargs={"filepath": downloaded_files["validation"][0]}),
                datasets.SplitGenerator(name=datasets.Split.TEST,
                                        gen_kwargs={"filepath": downloaded_files["test"][0]})
            ]
        else:
            return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                        gen_kwargs={"filepath": downloaded_files["train"][0]}),
                datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                        gen_kwargs={"filepath": downloaded_files["validation"][0]}),
            ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(wikitext): Yields (key, example) tuples from the dataset
        if filepath.endswith('.pkl'):
            data = pickle.load(open(os.path.abspath(filepath), "rb"))
        if filepath.endswith('.json'):
            data = json.load(open(os.path.abspath(filepath), "r"))
        if filepath.endswith('.txt'):
            data = open(os.path.abspath(filepath), "r").readlines()

        if filepath.endswith('.pkl') or filepath.endswith('.json'):
            for idx, doc in enumerate(data):
                if doc:
                    doc_tokens = [tok for d in doc['Sents'] for tok in d]
                    doc_ents_tags = ["O" for _ in range(len(doc_tokens))]
                    for ent in doc['Entities']:
                        # linked_entities = tuple(sorted(ent['linked_umls_entities'].items(), key=lambda x:x[1]['score'], reverse=True))
                        linked_entities = tuple(ent['linked_umls_entities'].items())
                        try:
                            tag = linked_entities[0][1]['type']
                            start, end = ent['pos'][0], ent['pos'][1]
                            ent_tags = [f"I-{tag}" for i in range(start, end)]
                            ent_tags[0] = f"B-{tag}"
                            if all(i in _LABELS for i in ent_tags):
                                doc_ents_tags[start:end] = ent_tags
                        except IndexError:
                            pass
                    doc_text = " ".join(doc_tokens)
                    yield idx, {"text": doc_text,"tokens": doc_tokens, "ner_tags": doc_ents_tags}
                else:
                    yield idx, {"text":"", "tokens": [], "ner_tags": []}

        elif filepath.endswith('.txt'):
            current_tokens, current_labels = [], []
            sentence_counter = 0
            for row in data:
                row = row.rstrip()
                if row:
                    token, label = row.split(" ")
                    current_tokens.append(token)
                    current_labels.append(label)
                else:
                    if not current_tokens:
                        continue
                    assert len(current_tokens) == len(current_labels), "💔 between len of tokens & labels"
                    doc_text = " ".join(current_tokens)
                    sentence = (sentence_counter, {"text": doc_text, "tokens": current_tokens, "ner_tags": current_labels})
                    sentence_counter += 1
                    current_tokens = []
                    current_labels = []
                    yield sentence
            # Don't forget last sentence in dataset 🧐
            if current_tokens:
                doc_text = " ".join(current_tokens)
                yield sentence_counter, {"text": doc_text, "tokens": current_tokens, "ner_tags": current_labels}
