from .base_llm import BaseLLM
from .data import Dataset, benchmark

import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm


def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Mask prompt portion
    labels = [-100] * question_len + input_ids[question_len:]
    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Round answer and wrap in <answer> tags.
    Example:
        Input: ("How many cm in one meter?", 100)
        Output: {"question": "How many cm in one meter?", "answer": "<answer>100</answer>"}
    """
    try:
        ans = round(float(answer), 6)
    except Exception:
        ans = answer

    question = f"Question: {prompt}\nRespond only with a number in the format <answer>value</answer>."
    formatted_answer = f"<answer>{ans}</answer>"
    return {"question": question, "answer": formatted_answer}


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formatted = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formatted)


def train_model(output_dir: str, num_train_epochs: int = 10, batch_size: int = 8, lr: float = 1e-3):
    """
    Fine-tune the model using LoRA adapters on the provided dataset.
    Saves trained adapters to `output_dir`.
    """
    trainset = Dataset("train")
    valset = Dataset("valid")

    base_llm = BaseLLM()
    model = base_llm.model
    tokenizer = base_llm.tokenizer

    # Prepare model for LoRA fine-tuning
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.01,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    train_data = TokenizedDataset(tokenizer, trainset, format_example)
    val_data = TokenizedDataset(tokenizer, valset, format_example)

    # New-style TrainingArguments (transformers >=4.50)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=lr,
        logging_steps=20,
        save_strategy="epoch",  # still valid
        eval_strategy="epoch",  # correct key for 4.52.4
        warmup_ratio=0.03,
        fp16=torch.cuda.is_available(),
        report_to=[],  # disables unwanted integrations (wandb, etc.)
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training complete.")
    test_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    llm.model.eval()

    result = benchmark(llm, testset, 100)
    print(f"accuracy={result.accuracy:.3f}  answer_rate={result.answer_rate:.3f}")


if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})
