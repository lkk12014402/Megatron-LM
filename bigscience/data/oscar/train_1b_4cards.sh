
export HABANA_VISIBLE_DEVICES=6,7,1,4
export HABANA_VISIBLE_MODULES=4,5,6,7

# BF16 config
HL_DATA_DIR_ROOT=/lkk/en_tokenized_merged \
HL_DATA_FILE_PREFIX=tokenized_text_document \
HL_TOKENIZER_MODEL=/lkk/Llama-3.1-8B/original/tokenizer.model \
HL_TOKENIZER_TYPE=Llama3Tokenizer \
HL_CKP_ACT=2 \
HL_LLAMA_VER=3.2 \
HL_DEVICES_PER_NODE=4 \
HL_DP=2 \
HL_TP=2 \
HL_PP=1 \
HL_LLAMA_MODEL_SIZE=1 \
/lkk/Megatron-LM/examples/llama/pretrain_llama.sh
