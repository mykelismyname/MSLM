#  <p align=center>`MSLM`</p>

#### Domain Sensitive Fine-tuning: 
Improving PLM sensitivity via Mask Specific Loss 

## Requirements

* Python 3.8+
* transformers 4.31.0
* torch 2.0.1

## Data
[BLURB](https://huggingface.co/datasets/EMBO/BLURB) benchmark dataset

### Data Preparation

```
python utils.py \
    [path to data] \
    [storage or destination directory]
```
Alternatively inherit pre-processed BLURB datasets such as,
* [BLURB](https://microsoft.github.io/BLURB/sample_code/data_generation.tar.gz)


### Masking
#### Our proposed Joint ELM-BLM masking approach
<img src="mslm_masking.png">

#### [PMI](https://github.com/AI21Labs/pmi-masking) masking
```
Construct a vocabularly from a dataset using the masking approach 

 ./run_pmi.sh
```


## Fine-tuning
```
Specify the paths to the data and set the masking budgets for both the Base level masking BLM and the Entity level masking ELM

./run_train.sh [DATASET]
```

## Citation

> Coming soon (NAACL 2024)