from .base_llm import BaseLLM
from .data import Dataset, benchmark


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

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    q = prompt
    if "<answer>" in q:
        q = q.replace("<answer>", "").replace("</answer>", "")
    q = q.strip()

    # --- STEP 2: Clean raw model output ---
    raw = str(answer).strip()

    # Keep only the first line before any HTML, PHP, etc.
    if "\n" in raw:
        raw = raw.split("\n")[0].strip()

    # --- STEP 3: Extract between <answer> and </answer> (manual, no regex) ---
    start = raw.find("<answer>")
    end = raw.find("</answer>")

    if start != -1 and end != -1 and end > start:
        val = raw[start + len("<answer>"):end].strip()
    else:
        # If tags missing, try to get first number-looking sequence
        val = ""
        for ch in raw:
            if ch.isdigit() or ch in ".-":
                val += ch
            elif val:
                break
        val = val.strip()

    # --- STEP 4: Convert numeric strings to clean rounded form ---
    cleaned = val
    try:
        num = float(val)
        cleaned = f"{round(num, 4):.4f}".rstrip("0").rstrip(".")
    except Exception:
        pass

    # --- STEP 5: Rebuild robust format ---
    formatted_question = f"Question: {q}\nAnswer:"
    formatted_answer = f"<answer>{cleaned}</answer>\n<end>"

    return {"question": formatted_question, "answer": formatted_answer}


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str,
    **kwargs,
):
    import torch
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model
    from pathlib import Path
    from .data import Dataset
    from .base_llm import BaseLLM

    llm = BaseLLM()

    # Apply LoRA adapters
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj","v_proj","k_proj","o_proj"],  # safer defaults for transformers
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm.model = get_peft_model(llm.model, lora_config)

    if torch.cuda.is_available():
        llm.model.enable_input_require_grads()

    # Load datasets
    train_data = Dataset("train")
    val_data = Dataset("valid")

    tokenized_train = TokenizedDataset(llm.tokenizer, train_data, format_example)
    tokenized_val = TokenizedDataset(llm.tokenizer, val_data, format_example)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        learning_rate=5e-5,
        gradient_checkpointing=True,
        save_strategy="epoch",
        eval_strategy="epoch",  
        logging_steps=10,
        remove_unused_columns=False,
        report_to=[],  # disable wandb/tensorboard auto-reporting
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=llm.tokenizer, mlm=False, mlm_probability=0.0)

    trainer = Trainer(
        model=llm.model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    trainer.train()

    # Save under 'sft_model' subfolder
    #save_path = Path(output_dir) / "sft_model"
    #save_path.mkdir(parents=True, exist_ok=True)
    #llm.model.save_pretrained(save_path)
    # Save directly into the output directory
    llm.model.save_pretrained(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
