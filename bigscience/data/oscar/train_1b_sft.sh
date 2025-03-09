# BF16 config
HL_DATA_DIR_ROOT=/lkk/Megatron-LM/examples/finetuning/tokenized_Sky-T1 \
HL_DATA_FILE_PREFIX=tokenized_Sky-T1_data_17k_text_document \
HL_TOKENIZER_MODEL=/lkk/Llama-3.1-8B/original/tokenizer.model \
HL_TOKENIZER_TYPE=Llama3Tokenizer \
HL_CKP_ACT=2 \
HL_LLAMA_VER=3.2 \
HL_DEVICES_PER_NODE=1 \
HL_GBS=1 \
HL_SEQ_LEN=8192 \
HL_TRAIN_ITERS=10 \
HL_DP=1 \
HL_TP=1 \
HL_PP=1 \
HL_LLAMA_MODEL_SIZE=1 \
/lkk/Megatron-LM/examples/llama/sft_llama.sh
