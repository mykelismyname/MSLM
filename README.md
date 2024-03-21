#  <p align=center>`MSLM`</p>

#### Domain Sensitive Fine-tuning: 
Improving PLM sensitivity via Mask Specific Loss 

## Requirements

* Python 3.8+
* transformers 4.31.0
* torch 2.0.1

## Data
[BLURB](https://huggingface.co/datasets/EMBO/BLURB) benchmark dataset

### Masking
> * Construct a vocabularly from a dataset using the [PMI](https://github.com/AI21Labs/pmi-masking) masking approach \
> ./run_pmi.sh


## Fine-tuning
> ./run_train.sh [DATASET]

## Citation

> Coming soon (NAACL 2024)