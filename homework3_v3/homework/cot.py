from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        chat = [
            {
                "role": "system",
                "content": (
                    "You are a precise math tutor. "
                    "Convert between measurement units carefully (meters, feet, yards, miles, inches, etc). "
                    "Show one brief reasoning step. "
                    "At the end, give ONLY the numeric result, with no units or text, "
                    "inside <answer></answer> tags. "
                    "Do not include spaces or words outside the tags."
                ),
            },
            {
                "role": "user",
                "content": "Convert 3 meters to feet.",
            },
            {
                "role": "assistant",
                "content": "1 meter = 3.28084 feet. 3 × 3.28084 = 9.84252. <answer>9.84252</answer>",
            },
            {
                "role": "user",
                "content": "Convert 5 feet to inches.",
            },
            {
                "role": "assistant",
                "content": "1 foot = 12 inches. 5 × 12 = 60. <answer>60</answer>",
            },
            {
                "role": "user",
                "content": "Convert 2 miles to yards.",
            },
            {
                "role": "assistant",
                "content": "1 mile = 1760 yards. 2 × 1760 = 3520. <answer>3520</answer>",
            },
            {
                "role": "user",
                "content": "Convert 36 inches to feet.",
            },
            {
                "role": "assistant",
                "content": "1 foot = 12 inches. 36 ÷ 12 = 3. <answer>3</answer>",
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
