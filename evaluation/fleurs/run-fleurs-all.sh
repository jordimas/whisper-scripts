#!/bin/bash

declare -a arr=("openai/whisper-medium" "softcatala/whisper-medium-ca" "jordimas/whisper-medium-ca-200steps" "jordimas/whisper-medium-ca-2000steps" "jordimas/whisper-medium-ca-5000steps")

for model in "${arr[@]}"; do
    echo Processing $model
    python3 ../run_eval_whisper_streaming.py --model_id="$model" --dataset="google/fleurs" --config="ca_es" --device=0 --language="ca"
done

