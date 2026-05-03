import os
from transformers import pipeline, GenerationConfig
from dotenv import load_dotenv


def load_generator():
    load_dotenv(".secrets")
    token = os.getenv("HF_TOKEN")
    return pipeline("text-generation", model="gpt2", token=token)


def greedy_decoding(gen):
    config = GenerationConfig(max_new_tokens=100)
    result = gen("The goblin keep training daily in order to", generation_config=config)
    print(result[0]["generated_text"])


def sampling(gen):
    prompt = "The goblin keep training daily in order to"

    config_confident = GenerationConfig(
        max_new_tokens=100,
        do_sample=True,
        temperature=0.5,
        top_k=50,
    )
    config_random = GenerationConfig(
        max_new_tokens=100,
        do_sample=True,
        temperature=1.5,
        top_k=50,
    )

    print("=== confident (temperature=0.5) ===")
    print(gen(prompt, generation_config=config_confident)[0]["generated_text"])

    print("\n=== random (temperature=1.5) ===")
    print(gen(prompt, generation_config=config_random)[0]["generated_text"])


def multiple_sequences(gen):
    config = GenerationConfig(
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        num_return_sequences=3,
    )
    results = gen("The goblin keep training daily in order to", generation_config=config)
    for i, r in enumerate(results, 1):
        print(f"\n--- Sequence {i} ---")
        print(r["generated_text"])


if __name__ == "__main__":
    generator = load_generator()
    greedy_decoding(generator)
    sampling(generator)
    multiple_sequences(generator)
