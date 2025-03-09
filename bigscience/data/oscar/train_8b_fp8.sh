# FP8 config
HL_DATA_DIR_ROOT=/lkk/en_tokenized_merged \
HL_DATA_FILE_PREFIX=tokenized_text_document \
HL_TOKENIZER_MODEL=/lkk/Llama-3.1-8B/original/tokenizer.model \
HL_FP8=1 \
HL_TRANSFORMER_IMPL=transformer_engine \
HL_SEQ_PARALLEL=0 \
HL_TOKENIZER_TYPE=Llama3Tokenizer \
HL_CKP_ACT=2 \
HL_LLAMA_VER=3.1 \
HL_DP=4 \
HL_TP=2 \
HL_PP=1 \
HL_LLAMA_MODEL_SIZE=8 \
/lkk/Megatron-LM/examples/llama/pretrain_llama.sh
