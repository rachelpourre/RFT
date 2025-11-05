from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        chat = [
            {
                "role": "system",
                "content": (
                    "You are a math assistant. "
                    "Solve simple word problems. "
                    "Give only the final numeric answer inside <answer></answer> tags. "
                    "Do not include any explanation or extra words."
                ),
            },
            {
                "role": "user",
                "content": "If a car travels 150 miles in 3 hours, what is its speed?",
            },
            {
                "role": "assistant",
                "content": "<answer>50</answer>",
            },
            {
                "role": "user",
                "content": question,
            },
        ]
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
