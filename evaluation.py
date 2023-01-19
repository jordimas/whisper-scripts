from huggingface_hub import login
from datasets import load_dataset
from transformers import pipeline
import evaluate
import pandas as pd
from datasets import Audio

def load_datasets():

    common_voice = load_dataset("mozilla-foundation/common_voice_11_0", "en", revision="streaming", split="test", streaming=True, use_auth_token=True)

    esb_datasets = {
        "Common Voice": common_voice,
    }
    
    return esb_datasets

def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    else:
        raise ValueError(f"Sample: {sample.keys()} has no transcript.")


whisper_asr = pipeline(
    "automatic-speech-recognition", model="openai/whisper-tiny.en")

        
whisper_norm = whisper_asr.tokenizer._normalize

def normalise(batch):
    batch["norm_text"] = whisper_norm(get_text(batch))
    return batch
    

def is_target_text_in_range(ref):
    ref = ref.strip()
    return ref not in filter_sequences    
    
def data(dataset):
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference": item["norm_text"]}
        
        
def main():



    wer_metric = evaluate.load("wer")
        
    filter_sequences = ["ignore time segment in scoring", ""]

    # set the batch size in accordance to your device
    BATCH_SIZE = 1
    wer_results = []

    # loop over all the datasets in the ESB benchmark
    esb_datasets = load_datasets()
    for dataset_name, dataset in esb_datasets.items():
        # only for debugging, restricts the number of rows to numeric value in brackets
        dataset = dataset.take(128)

        # resample to 16kHz
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        # normalise references
        dataset = dataset.map(normalise)

        # remove any empty references
        dataset = dataset.filter(is_target_text_in_range, input_columns=["norm_text"])

        # placeholders for predictions and references
        predictions = []
        references = []

        # run streamed inference
        for out in whisper_asr(data(dataset)):
            predictions.append(whisper_norm(out["text"]))
            references.append(out["reference"][0])

        # compute the WER
        wer = wer_metric.compute(references=references, predictions=predictions)
        wer = round(100 * wer, 2)

        wer_results.append(wer)        


    df = pd.DataFrame({"Dataset": esb_datasets.keys(), "WER": wer_results})
    print(df)


if __name__ == "__main__":
    main()

