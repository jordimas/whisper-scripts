#!/bin/bash

declare -a arr=("openai/whisper-medium" "softcatala/whisper-medium-ca" "openai/whisper-small" "softcatala/whisper-small-ca")

for model in "${arr[@]}"; do
    echo Processing $model
    python3 run_eval_whisper_streaming.py --model_id="$model" --dataset="google/fleurs" --config="ca_es" --device=0 --language="ca"
    python3 run_eval_whisper_streaming.py --model_id="$model" --dataset="projecte-aina/parlament_parla" --config="clean" --device=0 --language="ca"
    python3 run_eval_whisper_streaming.py --model_id="$model" --config ca --language ca --dataset mozilla-foundation/common_voice_11_0 --device 0
done

