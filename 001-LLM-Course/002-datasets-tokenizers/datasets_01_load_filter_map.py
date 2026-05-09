from datasets import load_dataset
from transformers import AutoTokenizer


def load_data():
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return dataset, tokenizer


def inspect_dataset(dataset):
    print(dataset)
    print(dataset["train"][0])
    print(dataset["train"].features)


def filter_rows(dataset):
    short = dataset["train"].filter(lambda x: len(x["text"]) < 500)
    print(f"There are {len(short)} rows in the training split after filtering.")
    print("--------------------------------------------------------")
    print(short[0])
    return short


def tokenize_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def verify_tokenized(tokenized):
    print(tokenized["train"][0])
    print(len(tokenized["train"][0]["input_ids"]))
    print(len(tokenized["train"][0]["attention_mask"]))


def set_pytorch_format(tokenized):
    tokenized["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    print(tokenized["train"][0])


if __name__ == "__main__":
    ds, tknzr = load_data()
    inspect_dataset(ds)
    filter_rows(ds)
    ds_tokenized = tokenize_dataset(ds, tknzr)
    verify_tokenized(ds_tokenized)
    set_pytorch_format(ds_tokenized)
