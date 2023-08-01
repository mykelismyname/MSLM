"""TODO(wikitext): Add a description here."""


import os
import pickle
import datasets

_DESCRIPTION = """\
 A biomedical corpus containing Electronic Health Records (EHRs).
"""
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""
_DATA_URL = ""


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
            features=["text"],
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
                    "text": datasets.Value("string")
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
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            for idx, doc in enumerate(data):
                if doc:
                    doc_text = [tok for d in doc['Sents'] for tok in d]
                    doc_text = " ".join(doc_text)
                    yield idx, {"text": doc_text}
                else:
                    yield idx, {"text": ""}


# with open('../data/dev.pkl', "rb") as f, open('../data/dev.txt', 'w') as g:
#     data = pickle.load(f)
#     for idx, doc in enumerate(data):
#         if doc:
#             doc_text = [tok for d in doc['Sents'] for tok in d]
#             doc_text = " ".join(doc_text)
#             g.write("{}\n".format(doc_text))