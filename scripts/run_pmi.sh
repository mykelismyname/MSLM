#!/bin/bash
export MIN_FREQ="2,2,2"
export DATA_PATH=../../WeLT/datasets/NER
export DATASET_NAME=NCBI-disease
export PRE_TRAINED_MODEL=dmis-lab/biobert-v1.1
export DATA_LOADING_SCRIPT=../mslm/ner_dataload.py

CUDA_VISIBLE_DEVICES=0 \
python3 ../mslm/masking/pmi_masking.py \
  --min_freq ${MIN_FREQ} \
  --data_path ${DATA_PATH}/${DATASET_NAME} \
  --dataset_name ${DATASET_NAME} \
  --model ${PRE_TRAINED_MODEL} \
  --loading_dataset_script ${DATA_LOADING_SCRIPT}
