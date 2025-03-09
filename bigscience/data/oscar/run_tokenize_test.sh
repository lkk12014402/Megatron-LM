# loop count will change based on number of files for a given dataset
mkdir en_tokenized_test
for i in $(seq 4 4);
do
  python /lkk/Megatron-LM/tools/preprocess_data.py --input test_en/oscar-${i}.jsonl --output-prefix en_tokenized_test/tokenized${i} --tokenizer-type Llama3Tokenizer --tokenizer-model /lkk/Llama-3.1-8B/original/tokenizer.model --append-eod --workers 16 --partitions 1
  exit
done
