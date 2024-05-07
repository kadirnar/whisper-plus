import torch
from hqq.utils.patching import prepare_for_inference
from pipelines.whisper import SpeechToTextPipeline
from transformers import BitsAndBytesConfig, HqqConfig
from utils.download_utils import download_youtube_to_mp3

url = "https://www.youtube.com/watch?v=BpN4hEAvDBg"
audio_path = download_youtube_to_mp3(url)

hqq_config = HqqConfig(
    nbits=1,
    group_size=64,
    quant_zero=False,
    quant_scale=False,
    axis=0,
    offload_meta=False,
)  # axis=0 is used by default

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = SpeechToTextPipeline(
    model_id="distil-whisper/distil-large-v3", quant_config=hqq_config)  # or bnb_config

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
transcript = model(
    audio_path=audio_path,
    chunk_length_s=30,
    stride_length_s=5,
    max_new_tokens=128,
    batch_size=100,
    language="english",
    return_timestamps=False)
end_event.record()

torch.cuda.synchronize()
elapsed_time_ms = start_event.elapsed_time(end_event)
seconds = elapsed_time_ms / 1000
print(f"Elapsed time: {seconds} seconds")
