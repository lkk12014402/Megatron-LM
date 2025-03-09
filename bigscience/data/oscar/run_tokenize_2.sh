# loop count will change based on number of files for a given dataset
mkdir en_tokenized
for i in $(seq 0 4);
do
  python /data1/lkk/Megatron-LM/tools/preprocess_data.py --input en/oscar-${i}.jsonl --output-prefix en_tokenized/tokenized${i} --tokenizer-type Llama3Tokenizer --tokenizer-model /data1/lkk/Llama-3.1-8B/original/tokenizer.model --append-eod --workers 32
done
