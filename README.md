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

# 🤗 Model Hub

You can find the models on the [HuggingFace Spaces](https://huggingface.co/spaces/ArtGAN/WhisperPlus) or on the [HuggingFace Model Hub](https://huggingface.co/models?search=whisper)

## 🎙️ Usage

To use the whisperplus library, follow the steps below for different tasks:

### 🎵 Youtube URL to Audio

```python
from whisperplus import SpeechToTextPipeline, download_and_convert_to_mp3

url = "https://www.youtube.com/watch?v=di3rHkEZuUw"
video_path = download_and_convert_to_mp3(url)
pipeline = SpeechToTextPipeline(model_id="openai/whisper-large-v3")
transcript = pipeline(
    audio_path=video_path, model_id="openai/whisper-large-v3", language="english
)

return transcript
```

### Contri

```bash
pip install -r dev-requirements.txt
pre-commit install
pre-commit run --all-files
```

## 📜 License

This project is licensed under the terms of the Apache License 2.0.

## 🤗 Acknowledgments

This project is based on the [HuggingFace Transformers](https://github.com/huggingface/transformers) library.

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
