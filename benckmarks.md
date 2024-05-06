This evaluation was performed on the Mozilla-Foundation/Common-Voice-17-0 dataset, a widely-used benchmark for speech-to-text models.

| Model                                | Metric Value |
| ------------------------------------ | ------------ |
| distil-whisper/distil-large-v3 + Hqq | 120.88       |
| distil-whisper/distil-large-v3       | 120.48       |
| distil-whisper/distil-large-v3 + Bnb | 120.14       |

Bnb: https://github.com/TimDettmers/bitsandbytes

Dataset: https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0

Hqq: https://github.com/mobiusml/hqq/
