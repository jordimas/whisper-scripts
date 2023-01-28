declare -a arr=("openai/whisper-small" "softcatala/whisper-small-ca"  "openai/whisper-medium" "softcatala/whisper-medium-ca" "jordimas/whisper-medium-ca-200steps" "jordimas/whisper-medium-ca-2000steps" "jordimas/whisper-medium-ca-5000steps")

for model in "${arr[@]}"; do
    echo Processing $model
    python3 ../run_eval_whisper_streaming.py --model_id="$model" --dataset="projecte-aina/parlament_parla" --config="clean" --device=0 --language="ca"
done
