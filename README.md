<div align="center">
<h2>
    WhisperPlus: Advancing Speech2Text and Text2Speech Processing 🚀
</h2>
<div>
    <img width="500" alt="teaser" src="doc\openai-whisper.jpg">
</div>
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
pip install whisperplus
```

## 🤗 Model Hub

You can find the models on the [HuggingFace Model Hub](https://huggingface.co/models?search=whisper)

## 🎙️ Usage

To use the whisperplus library, follow the steps below for different tasks:

### 🎵 Youtube URL to Audio

```python
from whisperplus import SpeechToTextPipeline, download_and_convert_to_mp3

url = "https://www.youtube.com/watch?v=di3rHkEZuUw"

audio_path = download_and_convert_to_mp3(url)
pipeline = SpeechToTextPipeline(model_id="openai/whisper-large-v3")
transcript = pipeline(audio_path, "openai/whisper-large-v3", "english")

print(transcript)
```

### 📰 Summarization

```python
from whisperplus import TextSummarizationPipeline

summarizer = TextSummarizationPipeline(model_id="facebook/bart-large-cnn")
summary = summarizer.summarize(transcript)
print(summary[0]["summary_text"])
```

### 📰 Long Text Support Summarization

```python
from whisperplus import LongTextSummarizationPipeline

summarizer = LongTextSummarizationPipeline(model_id="facebook/bart-large-cnn")
summary_text = summarizer.summarize(transcript)
print(summary_text)
```

### 💬 Speaker Diarization

```python
from whisperplus import (
    ASRDiarizationPipeline,
    download_and_convert_to_mp3,
    format_speech_to_dialogue,
)

audio_path = download_and_convert_to_mp3("https://www.youtube.com/watch?v=mRB14sFHw2E")

device = "cuda"  # cpu or mps
pipeline = ASRDiarizationPipeline.from_pretrained(
    asr_model="openai/whisper-large-v3",
    diarizer_model="pyannote/speaker-diarization",
    use_auth_token=False,
    chunk_length_s=30,
    device=device,
)

output_text = pipeline(audio_path, num_speakers=2, min_speaker=1, max_speaker=2)
dialogue = format_speech_to_dialogue(output_text)
print(dialogue)
```

### ⭐ RAG - Chat with Video(LanceDB)

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
    input_file="audio.mp3",
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

### 🎙️ Speech to Text

```python
from whisperplus import TextToSpeechPipeline

tts = TextToSpeechPipeline(model_id="suno/bark")
audio = tts(text="Hello World", voice_preset="v2/en_speaker_6")
```

## 😍 Contributing

```bash
pip install -r dev-requirements.txt
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
