sudo apt-get update -y
sudo apt-get install python3-pip ffmpeg -y
sudo apt install python3-virtualenv -y
virtualenv -p /usr/bin/python3 python3-env
source python3-env/bin/activate
pip install -r requirements.txt

# Simple test
#python3 run_eval_whisper_streaming.py --model_id="openai/whisper-medium" --dataset="google/fleurs" --config="ca_es" --device=0 --language="ca"
