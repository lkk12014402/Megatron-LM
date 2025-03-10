# BF16 config
#HL_FP8=1 \
#HL_TRANSFORMER_IMPL=transformer_engine \
#HL_SEQ_PARALLEL=0 \
HL_LOAD_DIR=/lkk/Megatron-LM/bigscience/data/oscar/Llama-3.1-8B-Instruct-mcore \
HL_DATA_DIR_ROOT=/lkk/Megatron-LM/examples/finetuning/tokenized_Sky-T1 \
HL_DATA_FILE_PREFIX=tokenized_Sky-T1_data_17k_text_document \
HL_TOKENIZER_MODEL=/lkk/Llama-3.1-8B/original/tokenizer.model \
HL_TOKENIZER_TYPE=Llama3Tokenizer \
HL_CKP_ACT=2 \
HL_LLAMA_VER=3.1 \
HL_DEVICES_PER_NODE=8 \
HL_GBS=16 \
HL_MICRO_BATCH=2 \
HL_LOG_INTERVAL=1 \
HL_SEQ_LEN=8192 \
HL_TRAIN_ITERS=2000 \
HL_DP=2 \
HL_TP=4 \
HL_PP=1 \
HL_LLAMA_MODEL_SIZE=8 \
/lkk/Megatron-LM/examples/llama/sft_llama.sh
