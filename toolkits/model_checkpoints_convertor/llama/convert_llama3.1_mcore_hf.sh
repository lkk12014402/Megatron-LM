export PT_HPU_GPU_MIGRATION=1
python /lkk/Megatron-LM/tools/checkpoint/convert_mlm_to_hf_checkpoint.py \
    --ckpt-dir-name "iter_0002000" \
    --target-params-dtype "bf16" \
    --source-model-type "llama3.1" \
    --load-path "./out/llama3.1_8b/bf16_transformer_engine_default_nl32_hs4096_ffn14336_gb16_mb2_sp1_D2_T4_P1_devices8_20250310_0827/checkpoints" \
    --save-path "sft_sky_llama3.1_8b_instruct_hf_checkpoints/"
