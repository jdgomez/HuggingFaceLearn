import os

from transformers import pipeline
from dotenv import load_dotenv

load_dotenv(".secrets")
hf_token = os.getenv("HF_TOKEN")


def load_classifier():
    load_dotenv(".secrets")
    token = os.getenv("HF_TOKEN")
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli", token=token)


def basic_example(clsf):
    text = "The new battery lasts 48 hours on a single charge."
    candidate_labels = ["technology", "sports", "cooking"]

    result = clsf(text, candidate_labels=candidate_labels)
    print(result)


def multi_label_mode(clsf):
    text_multilabel = "The athlete followed a strict diet and trained six hours a day to prepare for the Olympics."
    labels_multilabel = ["sports", "nutrition", "politics", "technology"]

    result_ml = clsf(text_multilabel, candidate_labels=labels_multilabel, multi_label=True)

    for label, score in zip(result_ml["labels"], result_ml["scores"]):
        print(f"{label:15s} {score:.3f}")


def batch_processing(clsf):
    texts = [
        "Scientists discover a new species of deep-sea fish.",
        "The central bank raised interest rates by 0.5%.",
        "The team scored three goals in the last ten minutes.",
    ]
    labels_batch = ["science", "economy", "sports", "politics"]

    results = clsf(texts, candidate_labels=labels_batch)

    for r in results:
        top_label = r["labels"][0]
        top_score = r["scores"][0]
        print(f"[{top_label} ({top_score:.2f})] {r['sequence'][:60]}...")


if __name__ == '__main__':
    classifier = load_classifier()
    basic_example(classifier)
    multi_label_mode(classifier)
    batch_processing(classifier)