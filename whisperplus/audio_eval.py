import torch
from datasets import load_dataset
from evaluate import load
from pipelines.whisper import SpeechToTextPipeline
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, BitsAndBytesConfig, HqqConfig, pipeline
from transformers.pipelines.pt_utils import KeyDataset

model_id = "distil-whisper/distil-large-v3"
processor = AutoProcessor.from_pretrained(model_id)


def hqq_load_model():
    from hqq.core.quantize import HQQBackend, HQQLinear
    from hqq.utils.patching import prepare_for_inference

    HQQLinear.set_backend(HQQBackend.PYTORCH)  # Pytorch backend
    HQQLinear.set_backend(HQQBackend.PYTORCH_COMPILE)  # Compiled Pytorch via dynamo
    HQQLinear.set_backend(
        HQQBackend.ATEN)  # C++ Aten/CUDA backend (set automatically by default if available)

    hqq_config = HqqConfig(
        nbits=4,
        group_size=64,
        quant_zero=False,
        quant_scale=False,
        axis=0,
        offload_meta=False,
    )  # axis=0 is used by default

    model = SpeechToTextPipeline(model_id="distil-whisper/distil-large-v3", quant_config=hqq_config)
    model = model.model
    return model


def base_load_model():
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        device="cuda:0",
    )

    return model


def calculate_metrics(model, processor, model_id):
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        torch_dtype=torch.bfloat16,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        model_kwargs={"use_flash_attention_2": True},
    )

    wer_metric = load("wer")

    common_voice_test = load_dataset(
        "mozilla-foundation/common_voice_17_0",  # mozilla-foundation/common_voice_17_0
        "dv",
        split="test")

    all_predictions = []

    # run streamed inference
    for prediction in tqdm(
            pipe(
                KeyDataset(common_voice_test, "audio"),
                max_new_tokens=128,
                generate_kwargs={"task": "transcribe"},
                batch_size=32,
            ),
            total=len(common_voice_test),
    ):
        all_predictions.append(prediction["text"])

    wer_ortho = 100 * wer_metric.compute(
        references=common_voice_test["sentence"], predictions=all_predictions)

    print(f"WER: {wer_ortho:.2f}%")
    return wer_ortho
