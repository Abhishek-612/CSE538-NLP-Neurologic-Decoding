#!/usr/bin/env bash

DATA_DIR='../dataset'
#DEVICES=$1
SPLIT=$1
MODEL_RECOVER_PATH=$2
OUTPUT_FILE=$3
BATCH_SIZE=1
BEAM_SIZE=6
#BATCH_SIZE=2
#BEAM_SIZE=2

#CUDA_VISIBLE_DEVICES=${DEVICES}
# run decoding
PYTHONPATH=../ python decode.py --model_name ${MODEL_RECOVER_PATH} \
  --input_path ${DATA_DIR}/lm/${SPLIT}.txt --output_file ${OUTPUT_FILE} \
  --constraint_file ${DATA_DIR}/clean/constraint/${SPLIT}.constraint.json \
  --batch_size ${BATCH_SIZE} --beam_size ${BEAM_SIZE} --max_tgt_length 200 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 1000 --sat_tolerance 2 --beta 0.1 #--early_stop 1.5
