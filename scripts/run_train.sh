#!/bin/bash
export SAVE_DIR="../output_"
#export DATASET_NAME="BioRED-Chem"
export PRE_TRAINED_MODEL=dmis-lab/biobert-v1.1
export BATCH_SIZE=8
export LOSS_REDUCTION='mean'
export MLM_PROB=0
export ELM_PROB=100
export STRATEGY="pmi"
export EPOCHS=20
export MAX_LENGTH=256
export META_EMBEDDING_SIZE=50

ner_datasets=("NCBI-disease" "BC5CDR-chem" "BC5CDR-disease" "BC2GM" "JNLPBA" "BioRED-Chem" "linnaeus")
ebm_datasets=("ebm-comet" "ebm-nlp")

TRAIN_SET="train.txt"
TEST_SET="test.txt"

if echo "${ner_datasets[@]}" | grep -qw "$1"; then
  DATA_LOADING_SCRIPT=../mslm/ner_dataload.py
  DATA_DIR=../../WeLT/datasets/NER
  DEV_SET="devel.txt"
  datasets=(${1})
elif echo "${ebm_datasets[@]}" | grep -qw "$1"; then
  DATA_LOADING_SCRIPT=../mslm/ebm_comet_dataload.py
  DATA_DIR=../data/
  DEV_SET="dev.txt"
  datasets=(${1})
elif [ "$1" = "pico" ]; then
  DATA_LOADING_SCRIPT=../mslm/ebm_comet_dataload.py
  DATA_DIR=../data/
  DEV_SET="dev.txt"
  datasets=("ebm-comet" "ebm-nlp")
elif [ "$1" = "ner" ]; then
  DATA_LOADING_SCRIPT=mslm/ner_dataload.py
  DATA_DIR=../WeLT/datasets/NER
  DEV_SET="devel.txt"
  datasets=("NCBI-disease" "BC5CDR-chem" "BC5CDR-disease" "BC2GM" "JNLPBA" "BioRED-Chem" "linnaeus")
elif [ "$1" = "mimic" ]; then
  DATA_LOADING_SCRIPT=../mslm/custom_dataload.py
  DATA_DIR=../data
  DEV_SET="dev.txt"
  MAX_LENGTH=256
  datasets=(${1})
fi

meta_emb_sizes=(100)
#base level (random) masking range
rm_start=0.075
rm_end=0.15
rm_step=0.075
#entity level masking range
em_start=0.0
em_end=1.0
em_step=0.25
x=(0.25)
y=(0.15 0.075 0.225)
models=("dmis-lab/biobert-v1.1" "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" "allenai/scibert_scivocab_uncased")
a=1
for DATASET_NAME in "${datasets[@]}";
  do
    for mlm__prob in "${y[@]}";
    do
      for elm__prob in "${x[@]}";
      do
        for m_emb_size in "${meta_emb_sizes[@]}";
        do
          for model in "${models[@]}";
          do
            echo "${a} DATASET - ${DATASET_NAME} BLM PROB - ${mlm__prob} STRATEGY ${STRATEGY} MODEL ${model}"
            CUDA_VISIBLE_DEVICES=0 \
            python3 run_train_no_tl.py \
            --model_name_or_path ${model} \
            --train_file ${DATA_DIR}/${DATASET_NAME}/${TRAIN_SET} \
            --validation_file ${DATA_DIR}/${DATASET_NAME}/${DEV_SET} \
            --test_file ${DATA_DIR}/${DATASET_NAME}/${TEST_SET} \
            --output_dir ${SAVE_DIR}/${DATASET_NAME} \
            --with_tracking \
            --pad_to_max_length \
            --task_name ner \
            --num_train_epochs ${EPOCHS} \
            --return_entity_level_metrics \
            --loading_dataset_script ${DATA_LOADING_SCRIPT} \
            --label_all_tokens \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --per_device_eval_batch_size ${BATCH_SIZE} \
            --reduction ${LOSS_REDUCTION} \
            --meta_embedding_dim ${m_emb_size} \
            --entity_masking \
            --random_mask \
            --max_length ${MAX_LENGTH} \
            --mlm_prob ${mlm__prob} \
            --elm_prob ${elm__prob} #pass a probability of 0 if you do not want base-level masking
          done
        done
      done
    done
  done
