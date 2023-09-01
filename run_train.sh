export SAVE_DIR=output_dir
export DATA_DIR=../WeLT/datasets/NER/BioRED-Chem
export PRE_TRAINED_MODEL=dmis-lab/biobert-v1.1
export BATCH_SIZE=8
export LOSS_REDUCTION='mean'
export MLM_PROB=0
export EPOCHS=4

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

CUDA_VISIBLE_DEVICES=0 \
python3 run_train_no_tl.py \
  --model_name_or_path ${PRE_TRAINED_MODEL} \
  --train_file ${DATA_DIR}/train.txt \
  --validation_file ${DATA_DIR}/devel.txt \
  --test_file ${DATA_DIR}/test.txt \
  --output_dir ${SAVE_DIR} \
  --with_tracking \
  --pad_to_max_length \
  --task_name ner \
  --num_train_epochs ${EPOCHS} \
  --return_entity_level_metrics \
  --loading_dataset_script mslm/ner_dataload.py \
  --label_all_tokens \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --reduction ${LOSS_REDUCTION} \
  --mlm_prob ${MLM_PROB} #pass a probability of 0 if you do not want base-level masking
#
#  --entity_masking \
#  --random_mask \
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