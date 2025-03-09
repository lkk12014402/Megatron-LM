#! /bin/bash

cmd="python build_idxmap_sft_dataset.py \
  --input /lkk/Sky-T1_data_17k.jsonl \
  --output-prefix tokenized_Sky-T1-test/tokenized_Sky-T1_data_17k \
  --tokenizer-type Llama3Tokenizer \
  --tokenizer-model /lkk/Llama-3.1-8B/original/tokenizer.model \
  --seq-length 8192 \
  --workers 1 \
  --partitions 1 --debug"

echo $cmd
eval $cmd

