from typing import Optional

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def load_model_whisper(
        model_id: str = "distil-whisper/distil-large-v3",
        quant_config=None,
        hqq_compile: bool = False,
        flash_attention_2: bool = False,
        device=None):
    """
    Loads a speech-to-text model and processor.

    Args:
    - model_id (str): The model ID to load (default: "distil-whisper/distil-large-v3").
    - quant_config: The quantization configuration (optional).
    - hqq_compile (bool): Whether to use HQQ compilation (default: False).
    - flash_attention_2 (bool): Whether to use flash attention 2 (default: False).
    - device: The device to use (e.g., "cuda" or "cpu").

    Returns:
    - The loaded model.
    """
    if hqq_compile:
        import hqq.models.base as hqq_base
        import torch._dynamo
        from hqq.core.quantize import HQQBackend, HQQLinear
        from hqq.models.hf.base import AutoHQQHFModel
        from hqq.utils.patching import prepare_for_inference

        torch._dynamo.config.suppress_errors = True

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

        processor = AutoProcessor.from_pretrained(model_id)
        HQQLinear.set_backend(HQQBackend.PYTORCH)

        AutoHQQHFModel.quantize_model(
            model.model.encoder, quant_config=quant_config, compute_dtype=torch.bfloat16, device=device)

        AutoHQQHFModel.quantize_model(
            model.model.decoder, quant_config=quant_config, compute_dtype=torch.bfloat16, device=device)

        hqq_base._QUANT_LAYERS = [torch.nn.Linear, HQQLinear]
        AutoHQQHFModel.set_auto_linear_tags(model.model.encoder)
        prepare_for_inference(model.model.encoder)

        AutoHQQHFModel.set_auto_linear_tags(model.model.decoder)
        prepare_for_inference(model.model.decoder, backend="torchao_int4")

        model.model.encoder.forward = torch.compile(
            model.model.encoder.forward, mode="reduce-overhead", fullgraph=True)
        model.model.decoder.forward = torch.compile(
            model.model.decoder.forward, mode="reduce-overhead", fullgraph=True)

    else:
        if flash_attention_2:
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "sdpa"

        import torch
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

        processor = AutoProcessor.from_pretrained(model_id)

    return model, processor
