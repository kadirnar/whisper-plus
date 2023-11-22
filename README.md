<div align="center">
<h2>
    WhisperPlus: Advancing Speech-to-Text Processing 🚀
</h2>
<div>
    <img width="500" alt="teaser" src="doc\openai-whisper.jpg">
</div>
<div>
    <a href="https://pypi.org/project/whisperplus" target="_blank">
        <img src="https://img.shields.io/pypi/pyversions/whisperplus.svg?color=%2334D058" alt="Supported Python versions">
    </a>
    <a href="https://badge.fury.io/py/whisperplus"><img src="https://badge.fury.io/py/whisperplus.svg" alt="pypi version"></a>
    <a href="https://huggingface.co/spaces/ArtGAN/WhisperPlus"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg" alt="HuggingFace Spaces"></a>
</div>
</div>

## 🛠️ Installation

```bash
pip install whisperplus
```

## 🎙️ Usage

To use the whisperplus library, follow the steps below for different tasks:

### 🎵 Youtube URL to Audio

```python
from whisperplus import SpeechToTextPipeline, download_and_convert_to_mp3

url = "https://www.youtube.com/watch?v=6Dh-RL__uN4"
video_path = download_and_convert_to_mp3(url)
pipeline = SpeechToTextPipeline()
transcript = pipeline(
    audio_path=video_path, model_id="openai/whisper-large-v3", language="turkish"
)

return transcript
```
