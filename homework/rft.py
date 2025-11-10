from .base_llm import BaseLLM
from .sft import train_model as sft_train_model, load as sft_load
from .sft import test_model as sft_test_model


def load() -> BaseLLM: #same as sft
    from pathlib import Path
    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    return llm


def train_model(output_dir: str, **kwargs):
    return sft_train_model(output_dir, **kwargs)


def test_model(ckpt_path: str):
    return sft_test_model(ckpt_path)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
