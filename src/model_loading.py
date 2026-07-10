# ==============================================================================
# Loading of the 4-bit quantized base model, with or without the LoRA adapter.
# Shared by training, inference and evaluation — platform-independent
# (secrets come from the environment, paths from arguments).
# ==============================================================================

import torch
from accelerate import dispatch_model, infer_auto_device_map
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def build_quantization_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantization with double quantization (could go 8-bit for more precision)."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(base_model_id: str):
    return AutoTokenizer.from_pretrained(base_model_id)


def load_base_model(base_model_id: str, device_map="auto", offload_dir=None,
                    max_memory=None):
    """Loads the quantized base model.

    With device_map="auto" the model is placed automatically (training setup).
    If max_memory is provided, the model is loaded on CPU first, a device map is
    inferred within the given budget and layers exceeding it are offloaded to
    offload_dir (constrained-GPU inference setup, e.g. a free Colab T4).
    """
    quantization_config = build_quantization_config()

    if max_memory is None:
        return AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map=device_map,
            quantization_config=quantization_config,
        )

    # loading model on CPU first for mapping
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )

    device_map = infer_auto_device_map(base_model, max_memory=max_memory)
    return dispatch_model(base_model, device_map=device_map, offload_dir=offload_dir)


def load_finetuned_model(base_model_id: str, adapter_path: str, offload_dir=None,
                         max_memory=None):
    """Loads the quantized base model and attaches the LoRA adapter, in eval mode."""
    base_model = load_base_model(
        base_model_id, offload_dir=offload_dir, max_memory=max_memory
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model
