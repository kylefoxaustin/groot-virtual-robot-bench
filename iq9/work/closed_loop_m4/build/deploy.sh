#!/bin/bash
# paths: set by setup/env.sh (or export them yourself)
: "${GVRB_WORK:=$HOME/gvrb}"
: "${IQ9_WORK:=/opt/gvrb}"
: "${IQ9_HOST:=iq9}"
set -euo pipefail
B=${IQ9_HOST}:${IQ9_WORK}/optionB
cd ${GVRB_WORK}/closed_loop_m4/build
ssh "$IQ9_HOST" "mkdir -p $IQ9_WORK/optionB/ctx_ft $IQ9_WORK/optionB/glue_ft $IQ9_WORK/optionB/io_ft"
echo "[deploy] bins $(date +%T)"
scp -q ctx/groot_vit_fp16.bin                 $B/ctx_ft/groot_vit_fp16.bin
scp -q ctx/groot_dit_fp16.bin                 $B/ctx_ft/groot_dit_fp16.bin
scp -q ctx/vlln_vl_self_attention_47_fp16.bin $B/ctx_ft/vlln_vl_self_attention_fp16.bin
scp -q ctx/state_encoder_47_fp16.bin          $B/ctx_ft/state_encoder_fp16.bin
scp -q ctx/action_encoder_47_fp16.bin         $B/ctx_ft/action_encoder_fp16.bin
scp -q ctx/action_decoder_47_fp16.bin         $B/ctx_ft/action_decoder_fp16.bin
echo "[deploy] glue $(date +%T)"
scp -q glue_constants/mlp1_0_weight.npy glue_constants/mlp1_0_bias.npy \
       glue_constants/ah_future_tokens_weight.npy glue_constants/ah_position_embedding_weight.npy \
       $B/glue_ft/
echo "[deploy] time_proj + golden $(date +%T)"
scp -q ah_fix/io/time_proj_t0.raw ah_fix/io/time_proj_t250.raw ah_fix/io/time_proj_t500.raw ah_fix/io/time_proj_t750.raw $B/io_ft/
scp -q ref_ft/host_e2e_ft.npy ref_ft/host_e2e_ft_null.npy $B/io_ft/
echo "[deploy] board-local copies (embed_tokens reuse + base io raws) $(date +%T)"
# IQ9_WORK is expanded here (host side); the {a,b,...} brace list is expanded by the board's shell
ssh "$IQ9_HOST" "cp $IQ9_WORK/optionB/glue_constants/embed_tokens_weight.npy $IQ9_WORK/optionB/glue_ft/ && cp $IQ9_WORK/optionB/io/{pixel_values,position_ids,input_ids,attention_mask,state,init_actions}.raw $IQ9_WORK/optionB/io_ft/ && echo BOARDCOPY_OK"
echo "[deploy] verify $(date +%T)"
ssh "$IQ9_HOST" "ls -la $IQ9_WORK/optionB/ctx_ft/ $IQ9_WORK/optionB/glue_ft/ $IQ9_WORK/optionB/io_ft/ | grep -E '\.bin|\.npy|\.raw|total'"
echo "DEPLOY_DONE"
