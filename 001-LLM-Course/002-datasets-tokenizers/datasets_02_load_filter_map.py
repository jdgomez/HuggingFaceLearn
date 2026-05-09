from datasets import load_dataset
from transformers import AutoTokenizer


def load_data():
    dataset = load_dataset("Eligemicasa/spain-real-estate-open-data", "municipalities")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return dataset, tokenizer


def inspect_dataset(dataset):
    print(dataset)
    print(dataset["train"][0])
    print(dataset["train"].features)


def filter_rows(dataset):
    priced = dataset["train"].filter(lambda x: x["price_avg"] is not None)
    print(f"There are {len(priced)} rows in the training split after filtering.")
    print("--------------------------------------------------------")
    print(priced[0])
    return priced


def tokenize_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(batch["province"], truncation=True, padding="max_length", max_length=128)

    return  dataset.map(tokenize, batched=True)


def verify_tokenized(tokenized):
    print(tokenized["train"][0])
    print(len(tokenized["train"][0]["input_ids"]))
    print(len(tokenized["train"][0]["attention_mask"]))


def set_pytorch_format(tokenized):
    tokenized["train"].set_format(type="torch", columns=["input_ids", "attention_mask"])
    print(tokenized["train"][0])


if __name__ == "__main__":
    ds, tknzr = load_data()
    inspect_dataset(ds)
    filter_rows(ds)
    ds_tokenized = tokenize_dataset(ds, tknzr)
    verify_tokenized(ds_tokenized)
    set_pytorch_format(ds_tokenized)
