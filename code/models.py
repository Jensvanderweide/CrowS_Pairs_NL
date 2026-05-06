import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Gated models that require a HuggingFace token
GATED_MODELS = {
    "Gemma-3-1b",
    "Llama-3.2-3B",
    "Llama-3.1-8B",
    "gemma-3-4b-it",
}

def load_model(model_name, device):
    """Load model and tokenizer from HuggingFace."""
    print("Loading model....")

    model_name_map = {
        "gpt2": "gpt2",
        "gpt2-medium": "openai-community/gpt2-medium",
        "EuroLLM1.7B": "utter-project/EuroLLM-1.7B",
        "EuroLLM9BInstruct": "utter-project/EuroLLM-9B-Instruct",
        "Gemma-3-1b": "google/gemma-3-1b-it",
        "Llama-3.2-3B": "meta-llama/Meta-Llama-3.2-3B",
        "deepseek1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "bloomz7b1-mt": "bigscience/bloomz-7b1-mt",
        "mistral7b-instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.1",
        "gemma-3-4b-it": "google/gemma-3-4b-it",
        "DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "GEITje-7B-ultra": "BramVanroy/GEITje-7B-ultra",
        "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    }

    if model_name not in model_name_map:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Log in to HuggingFace only for gated models
    if model_name in GATED_MODELS:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise EnvironmentError(
                f"Model '{model_name}' is gated and requires a HuggingFace token. "
                "Please set the HF_TOKEN environment variable."
            )
        login(token=hf_token)

    model_path = model_name_map[model_name]

    # Use bfloat16 on GPU to halve memory usage; float32 on CPU
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Ensure pad token is set (required for batched generation with causal LMs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Left-padding is required for correct batched causal LM generation
    tokenizer.padding_side = "left"

    # Load model directly onto the target device(s) with correct dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()

    print(f"Loaded {model_name} model and tokenizer successfully!")

    return tokenizer, model


def prepare_lm(model, tokenizer, device):
    """Prepare the language model dictionary."""
    return {
        "model": model,
        "tokenizer": tokenizer,
        "softmax": torch.nn.Softmax(dim=-1),
        "log_softmax": torch.nn.LogSoftmax(dim=-1),
        "mask_token": tokenizer.bos_token if hasattr(tokenizer, "bos_token") else tokenizer.cls_token,
        "uncased": False,
        "device": device,
    }
