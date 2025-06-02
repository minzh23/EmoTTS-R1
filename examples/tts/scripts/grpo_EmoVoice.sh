#!/bin/bash
export PYTHONPATH=$PYTHONPATH:~/EmoVoice/src
# export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1

code_dir=examples/tts
llm_path="/root/EmoVoice/checkpoint/Qwen2.5-0.5B"
codec_decoder_path="/root/EmoVoice/checkpoint/CosyVoice"
ckpt_path=/root/EmoVoice/checkpoint
split=test
val_data_path=/root/EmoVoice/EmoVoice-DB/test.jsonl

# vocabulary settings
code_layer=3            # 1 single semantic code layer   2 3 4 5 6 7 8 group semantic code layers 
total_audio_vocabsize=4160
total_vocabsize=156160  # 152000 + 4160 Sry: Here is not elegant to set the total_vocabsize manually, I may fix it later :)

# code settings
codec_decoder_type=CosyVoice
num_latency_tokens=0     # number of latency tokens (same as the number in training)
do_layershift=false      # if false, tokens in each layers use the same codebook, otherwise, use different codebooks

# model settings
group_decode=true
group_decode_adapter_type=linear

# decode config
text_repetition_penalty=1.2
audio_repetition_penalty=1.2        # default 1.0, set to 1.2 for reduce silence
max_new_tokens=3000                 # 3000 for CosyVoice-single
do_sample=true
top_p=1.0
top_k=0
temperature=1.0
decode_text_only=false
output_text_only=false
speech_sample_rate=22050

decode_log=$ckpt_path/tts_decode_${split}_rp${repetition_penalty}_seed${dataset_sample_seed}_greedy_kaiyuan

if [ "$decode_text_only" = true ] ; then
    decode_log=$decode_log"_text_only"
fi

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/grpo_tts.py \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name=qwen2.5-0.5b \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=896 \
        ++model_config.codec_decoder_path=$codec_decoder_path \
        ++model_config.codec_decode=true \
        ++model_config.vocab_config.code_layer=$code_layer \
        ++model_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
        ++model_config.vocab_config.total_vocabsize=$total_vocabsize \
        ++model_config.codec_decoder_type=$codec_decoder_type \
        ++model_config.group_decode=$group_decode \
        ++model_config.group_decode_adapter_type=$group_decode_adapter_type \
        ++model_config.use_text_stream=false \
        ++dataset_config.dataset=speech_dataset_tts \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.train_data_path=$val_data_path \
        ++dataset_config.inference_mode=true \
        ++dataset_config.vocab_config.code_layer=$code_layer \
        ++dataset_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
        ++dataset_config.vocab_config.total_vocabsize=$total_vocabsize \
        ++dataset_config.num_latency_tokens=$num_latency_tokens \
        ++dataset_config.do_layershift=$do_layershift \
        ++dataset_config.use_emo=true \
        ++train_config.model_name=tts \
        ++train_config.freeze_encoder=false \
        ++train_config.freeze_llm=false \
        ++train_config.freeze_group_decode_adapter=false \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.max_prompt_length=2048 \
        ++train_config.max_completion_length=768 \
        ++train_config.num_generations=4 \
        ++decode_config.text_repetition_penalty=$text_repetition_penalty \
        ++decode_config.audio_repetition_penalty=$audio_repetition_penalty \
        ++decode_config.max_new_tokens=$max_new_tokens \
        ++decode_config.do_sample=$do_sample \
        ++decode_config.top_p=$top_p \
        ++decode_config.top_k=$top_k \
        ++decode_config.temperature=$temperature \
        ++decode_config.decode_text_only=$decode_text_only \
        ++decode_config.num_latency_tokens=$num_latency_tokens \
        ++decode_config.do_layershift=$do_layershift \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/EmoVoice.pt \
        ++output_text_only=$output_text_only \
        ++speech_sample_rate=$speech_sample_rate \
        ++log_config.log_file=$decode_log/infer.log \

python examples/tts/utils/decode_whisper_v3.py --parent_dir $decode_log --audio_subdir pred_audio/neutral_prompt_speech

bash scripts/compute_wer.sh $decode_log

python examples/tts/utils/eval_emo_acc.py --gt $val_data_path --pred $decode_log --audio_subdir pred_audio/neutral_prompt_speech