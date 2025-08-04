<div align="center">
<h2>
    WhisperPlus: Faster, Smarter, and More Capable 🚀
</h2>
<div>
    <img width="500" alt="teaser" src="doc\openai-whisper.jpg">
</div>
    <a href="https://trendshift.io/repositories/6007" target="_blank"><img src="https://trendshift.io/api/badge/repositories/6007" alt="kadirnar%2Fwhisper-plus | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

<div>
    <a href="https://pypi.org/project/whisperplus" target="_blank">
        <img src="https://img.shields.io/pypi/pyversions/whisperplus.svg?color=%2334D058" alt="Supported Python versions">
    </a>
    <a href="https://badge.fury.io/py/whisperplus"><img src="https://badge.fury.io/py/whisperplus.svg" alt="pypi version"></a>
    <a href="https://huggingface.co/spaces/ArtGAN/Audio-WebUI"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg" alt="HuggingFace Spaces"></a>
</div>
</div>

## 🛠️ Installation

```bash
pip install whisperplus git+https://github.com/huggingface/transformers
pip install flash-attn --no-build-isolation
```

## 🤗 Model Hub

You can find the models on the [HuggingFace Model Hub](https://huggingface.co/models?search=whisper)

## 🎙️ Usage

To use the whisperplus library, follow the steps below for different tasks:

### 🎵 Youtube URL to Audio

```python
from whisperplus import SpeechToTextPipeline, download_youtube_to_mp3
from transformers import BitsAndBytesConfig, HqqConfig
import torch

url = "https://www.youtube.com/watch?v=di3rHkEZuUw"
audio_path = download_youtube_to_mp3(url, output_dir="downloads", filename="test")

hqq_config = HqqConfig(
    nbits=4,
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

pipeline = SpeechToTextPipeline(
    model_id="distil-whisper/distil-large-v3",
    quant_config=hqq_config,
    flash_attention_2=True,
)

transcript = pipeline(
    audio_path=audio_path,
    chunk_length_s=30,
    stride_length_s=5,
    max_new_tokens=128,
    batch_size=100,
    language="english",
    return_timestamps=False,
)

print(transcript)
```

### 🍎 Apple MLX

```python
from whisperplus.pipelines import mlx_whisper
from whisperplus import download_youtube_to_mp3

url = "https://www.youtube.com/watch?v=1__CAdTJ5JU"
audio_path = download_youtube_to_mp3(url)

text = mlx_whisper.transcribe(
    audio_path, path_or_hf_repo="mlx-community/whisper-large-v3-mlx"
)["text"]
print(text)
```

### 🍏 Lightning Mlx Whisper

```python
from whisperplus.pipelines.lightning_whisper_mlx import LightningWhisperMLX
from whisperplus import download_youtube_to_mp3

url = "https://www.youtube.com/watch?v=1__CAdTJ5JU"
audio_path = download_youtube_to_mp3(url)

whisper = LightningWhisperMLX(model="distil-large-v3", batch_size=12, quant=None)
output = whisper.transcribe(audio_path=audio_path)["text"]
```

### 📰 Summarization

```python
from whisperplus.pipelines.summarization import TextSummarizationPipeline

summarizer = TextSummarizationPipeline(model_id="facebook/bart-large-cnn")
summary = summarizer.summarize(transcript)
print(summary[0]["summary_text"])
```

### 📰 Long Text Support Summarization

```python
from whisperplus.pipelines.long_text_summarization import LongTextSummarizationPipeline

summarizer = LongTextSummarizationPipeline(model_id="facebook/bart-large-cnn")
summary_text = summarizer.summarize(transcript)
print(summary_text)
```

### 💬 Speaker Diarization

You must confirm the licensing permissions of these two models.

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

```bash
pip install -r requirements/speaker_diarization.txt
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

```python
from whisperplus.pipelines.whisper_diarize import ASRDiarizationPipeline
from whisperplus import download_youtube_to_mp3, format_speech_to_dialogue

audio_path = download_youtube_to_mp3("https://www.youtube.com/watch?v=mRB14sFHw2E")

device = "cuda"  # cpu or mps
pipeline = ASRDiarizationPipeline.from_pretrained(
    asr_model="openai/whisper-large-v3",
    diarizer_model="pyannote/speaker-diarization-3.1",
    use_auth_token=False,
    chunk_length_s=30,
    device=device,
)

output_text = pipeline(audio_path, num_speakers=2, min_speaker=1, max_speaker=2)
dialogue = format_speech_to_dialogue(output_text)
print(dialogue)
```

### ⭐ RAG - Chat with Video(LanceDB)

```bash
pip install sentence-transformers ctransformers langchain
```

```python
from whisperplus.pipelines.chatbot import ChatWithVideo

chat = ChatWithVideo(
    input_file="trascript.txt",
    llm_model_name="TheBloke/Mistral-7B-v0.1-GGUF",
    llm_model_file="mistral-7b-v0.1.Q4_K_M.gguf",
    llm_model_type="mistral",
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
)

query = "what is this video about ?"
response = chat.run_query(query)
print(response)
```

### 🌠 RAG - Chat with Video(AutoLLM)

```bash
pip install autollm>=0.1.9
```

```python
from whisperplus.pipelines.autollm_chatbot import AutoLLMChatWithVideo

# service_context_params
system_prompt = """
You are an friendly ai assistant that help users find the most relevant and accurate answers
to their questions based on the documents you have access to.
When answering the questions, mostly rely on the info in documents.
"""
query_wrapper_prompt = """
The document information is below.
---------------------
{context_str}
---------------------
Using the document information and mostly relying on it,
answer the query.
Query: {query_str}
Answer:
"""

chat = AutoLLMChatWithVideo(
    input_file="input_dir",  # path of mp3 file
    openai_key="YOUR_OPENAI_KEY",  # optional
    huggingface_key="YOUR_HUGGINGFACE_KEY",  # optional
    llm_model="gpt-3.5-turbo",
    llm_max_tokens="256",
    llm_temperature="0.1",
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    embed_model="huggingface/BAAI/bge-large-zh",  # "text-embedding-ada-002"
)

query = "what is this video about ?"
response = chat.run_query(query)
print(response)
```

### 🎙️ Text to Speech

```python
from whisperplus.pipelines.text2speech import TextToSpeechPipeline

tts = TextToSpeechPipeline(model_id="suno/bark")
audio = tts(text="Hello World", voice_preset="v2/en_speaker_6")
```

### 🎥 AutoCaption

```bash
pip install moviepy
apt install imagemagick libmagick++-dev
cat /etc/ImageMagick-6/policy.xml | sed 's/none/read,write/g'> /etc/ImageMagick-6/policy.xml
```

```python
from whisperplus.pipelines.whisper_autocaption import WhisperAutoCaptionPipeline
from whisperplus import download_youtube_to_mp4

video_path = download_youtube_to_mp4(
    "https://www.youtube.com/watch?v=di3rHkEZuUw",
    output_dir="downloads",
    filename="test",
)  # Optional

caption = WhisperAutoCaptionPipeline(model_id="openai/whisper-large-v3")
caption(video_path=video_path, output_path="output.mp4", language="english")
```

## 😍 Contributing

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## 📜 License

This project is licensed under the terms of the Apache License 2.0.

## 🤗 Citation

```bibtex
@misc{radford2022whisper,
  doi = {10.48550/ARXIV.2212.04356},
  url = {https://arxiv.org/abs/2212.04356},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  title = {Robust Speech Recognition via Large-Scale Weak Supervision},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
