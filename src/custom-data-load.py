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
_LABELS = ["O", "B-Clinical Drug", "B-Diagnostic Procedure", "B-Disease or Syndrome", "B-Finding",
"B-Injury or Poisoning", "B-Mental Process", "B-Mental or Behavioral Dysfunction", "B-Pathologic Function",
"B-Pharmacologic Substance", "B-Sign or Symptom", "B-Therapeutic or Preventive Procedure", "I-Clinical Drug",
"I-Diagnostic Procedure", "I-Disease or Syndrome", "I-Finding", "I-Injury or Poisoning", "I-Mental Process",
"I-Mental or Behavioral Dysfunction", "I-Pathologic Function", "I-Pharmacologic Substance", "I-Sign or Symptom",
"I-Therapeutic or Preventive Procedure"]

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
    """TODO(wikitext_103): Short description of my dataset."""

    # TODO(wikitext_103): Set up version.
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

        for idx, doc in enumerate(data):
            if doc:
                doc_tokens = [tok for d in doc['Sents'] for tok in d]
                doc_text = " ".join(doc_tokens)
                doc_ner_tags = []
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
                            if all(i in _LABELS for i in ent_tags):
                                sent_ents_tags[start:end] = ent_tags
                        except IndexError:
                            pass
                    doc_ner_tags.append(sent_ents_tags)
                doc_ner_tags = [t for e in doc_ner_tags for t in e]
                yield idx, {"text": doc_text, "tokens": doc_tokens, "ner_tags": doc_ner_tags}
            else:
                pass
                yield idx, {"text": "", "tokens": [], "ner_tags": []}

        # if filepath.endswith('.pkl'):
        #     with open(filepath, "rb") as f:
        #         data = pickle.load(f)
        #         for idx, doc in enumerate(data):
        #             if doc:
        #                 doc_tokens = [tok for d in doc['Sents'] for tok in d]
        #                 doc_text = " ".join(doc_tokens)
        #                 yield idx, {"text": doc_text, "tokens": doc_tokens}
        #             else:
        #                 yield idx, {"text": "", "tokens":doc_tokens}
