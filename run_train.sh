export SAVE_DIR=output_dir
export DATA_DIR=../WeLT/datasets/NER/BioRED-Chem
export PRE_TRAINED_MODEL=dmis-lab/biobert-v1.1

python3 run_train.py \
    --train_file ${DATA_DIR}/train.txt \
    --validation_file ${DATA_DIR}/devel.txt \
    --model_name_or_path ${PRE_TRAINED_MODEL} \
    --entity_masking \
    --label_all_tokens \
    --pad_to_max_length \
    --seed 42 \
    --num_train_epochs 20 \
    --detection \
    --loading_dataset_script mslm/ner_dataload.py \
    --max_seq_length 128 \
    --meta_embedding_dim 30 \
    --output_dir ${SAVE_DIR}