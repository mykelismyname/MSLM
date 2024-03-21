#!/bin/bash
export SAVE_DIR="output_"
#export DATASET_NAME="BioRED-Chem"
export PRE_TRAINED_MODEL=allenai/scibert_scivocab_uncased
export BATCH_SIZE=8
export LOSS_REDUCTION='mean'
export MLM_PROB=0
export ELM_PROB=100
export EPOCHS=20
export MAX_LENGTH=256
export META_EMBEDDING_SIZE=50

ner_datasets=("BC2GM" "BC5CDR-chem" "BC5CDR-disease" "NCBI-disease" "JNLPBA" "BioRED-Chem" "linnaeus")
ebm_datasets=("ebm-comet" "ebm-nlp")

TRAIN_SET="train.txt"
TEST_SET="test.txt"

if echo "${ner_datasets[@]}" | grep -qw "$1"; then
  DATA_LOADING_SCRIPT=mslm/ner_dataload.py
  DATA_DIR=../WeLT/datasets/NER
  DEV_SET="devel.txt"
  datasets=(${1})
elif echo "${ebm_datasets[@]}" | grep -qw "$1"; then
  DATA_LOADING_SCRIPT=mslm/ebm_comet_dataload.py
  DATA_DIR=data/
  DEV_SET="dev.txt"
  datasets=(${1})
elif [ "$1" = "pico" ]; then
  DATA_LOADING_SCRIPT=mslm/ebm_comet_dataload.py
  DATA_DIR=data/
  DEV_SET="dev.txt"
  datasets=("ebm-comet" "ebm-nlp")
elif [ "$1" = "ner" ]; then
  DATA_LOADING_SCRIPT=mslm/ner_dataload.py
  DATA_DIR=../WeLT/datasets/NER
  DEV_SET="devel.txt"
  datasets=("JNLPBA" "BC5CDR-disease" "BioRED-Chem" "BC5CDR-chem" "linnaeus")
elif [ "$1" = "mimic" ]; then
  DATA_LOADING_SCRIPT=mslm/custom_dataload.py
  DATA_DIR=data
  DEV_SET="dev.txt"
  MAX_LENGTH=256
  datasets=(${1})
fi

echo "Dataset loading script-" ${DATA_LOADING_SCRIPT}
echo "Data dir-" ${DATA_DIR}

meta_emb_sizes=(100)
#base level (random) masking range
rm_start=0.15
rm_end=0.225
rm_step=0.075
#entity level masking range
em_start=0.25
em_end=1.0
em_step=0.25

for DATASET_NAME in "${datasets[@]}";
  do
    echo "DATASET - ${DATASET_NAME}"
    for mlm__prob in $(seq $rm_start $rm_step $rm_end);
    do
      for elm__prob in $(seq $em_start $em_step $em_end);
      do
        for m_emb_size in "${meta_emb_sizes[@]}";
        do
          echo "DATASET - ${DATASET_NAME} BLM PROB - ${mlm__prob} ELM PROB - ${elm__prob}"
          CUDA_VISIBLE_DEVICES=0 \
          python3 run_train_no_tl.py \
            --model_name_or_path ${PRE_TRAINED_MODEL} \
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
#DATASET_NAME=$1

#python3 run_train.py \
#    --train_file ${DATA_DIR}/train.txt \
#    --validation_file ${DATA_DIR}/devel.txt \
#    --model_name_or_path ${PRE_TRAINED_MODEL} \
#    --entity_masking \
#    --label_all_tokens \
#    --pad_to_max_length \
#    --seed 42 \
#    --num_train_epochs 2 \
#    --detection \
#    --loading_dataset_script mslm/ner_dataload.py \
#    --max_seq_length 128 \
#    --meta_embedding_dim 30 \
#    --output_dir ${SAVE_DIR}

#python3 run_train_tl.py \
#  --model_name_or_path ${PRE_TRAINED_MODEL} \
#  --train_file ${DATA_DIR}/train.txt \
#  --validation_file ${DATA_DIR}/devel.txt \
#  --test_file ${DATA_DIR}/test.txt \
#  --output_dir ${SAVE_DIR} \
#  --pad_to_max_length \
#  --task_name ner \
#  --num_train_epochs 1 \
#  --return_entity_level_metrics \
#  --loading_dataset_script mslm/ner_dataload.py \
#  --do_train \
#  --do_eval \
#  --do_predict \
#  --entity_masking \
#  --overwrite_output_dir

#CUDA_VISIBLE_DEVICES=0 \
#    python3 run_train_no_tl.py \
#      --model_name_or_path ${PRE_TRAINED_MODEL} \
#      --train_file ${DATA_DIR}/${DATASET_NAME}/${TRAIN_SET} \
#      --validation_file ${DATA_DIR}/${DATASET_NAME}/${DEV_SET} \
#      --test_file ${DATA_DIR}/${DATASET_NAME}/${TEST_SET} \
#      --output_dir ${SAVE_DIR}/${DATASET_NAME} \
#      --with_tracking \
#      --pad_to_max_length \
#      --task_name ner \
#      --num_train_epochs ${EPOCHS} \
#      --return_entity_level_metrics \
#      --loading_dataset_script ${DATA_LOADING_SCRIPT} \
#      --label_all_tokens \
#      --per_device_train_batch_size ${BATCH_SIZE} \
#      --per_device_eval_batch_size ${BATCH_SIZE} \
#      --reduction ${LOSS_REDUCTION} \
#      --max_length ${MAX_LENGTH}