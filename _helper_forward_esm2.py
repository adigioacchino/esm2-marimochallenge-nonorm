import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")

with app.setup:
    import os
    import gzip
    import shutil
    import urllib.request
    import random
    from datasets import load_dataset

    import marimo as mo
    import torch
    from torch.utils.data import IterableDataset
    from transformers import (
        EsmTokenizer,
        EsmConfig,
        EsmForMaskedLM,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    from Bio import SeqIO


@app.cell
def _():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (device,)


@app.cell
def _(device):
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)

    # Send to device in eval mode
    model.to(device)
    model.eval()
    return model, tokenizer


@app.cell
def _(test_ratio):
    class StreamingUniRefDataset(IterableDataset):
        def __init__(self, tokenizer, split="train", max_length=1024, seed=1):
            self.tokenizer = tokenizer
            self.split = split
            self.test_ratio = test_ratio
            self.max_length = max_length
            self.seed = seed
            self.uniref_ds = load_dataset(
                "agemagician/uniref30", split=split, streaming=True
            ).shuffle(seed=seed)

        def __iter__(self):
            # Setting a seed helps keep the split consistent during a single training run
            random.seed(self.seed)

            for x in self.uniref_ds:
                protein_sequence = x["text"]

                # Process and yield the sequence
                encoding = self.tokenizer(
                    protein_sequence,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )

                yield {key: val.squeeze(0) for key, val in encoding.items()}

    return (StreamingUniRefDataset,)


@app.cell
def _(StreamingUniRefDataset, tokenizer):
    max_length = 100  # Limit sequences to 100 amino acids for faster training
    train_dataset = StreamingUniRefDataset(
        tokenizer=tokenizer,
        split="train",
        max_length=max_length,
    )

    # 15% of data streams here
    eval_dataset = StreamingUniRefDataset(
        tokenizer=tokenizer,
        split="test",
        max_length=max_length,
    )
    return eval_dataset, train_dataset


@app.cell
def _(eval_dataset, model, tokenizer, train_dataset):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # 6. Strict Hyperparameters for valid comparison
    training_args = TrainingArguments(
        output_dir="./esm2_comparison_run",
        max_steps=10,  # Fixed step limit ensures identical exposure to data
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        # Disk Space Optimization
        save_total_limit=2,
        # Evaluation strategy settings
        eval_strategy="steps",  # Required for streaming iterable datasets
        eval_steps=1,  # Calculates validation loss every 1000 steps
        logging_steps=1,
        save_steps=1,
        learning_rate=4e-4,
        weight_decay=0.01,
        fp16=True,
        seed=42,  # Keeps the Data Collator's random masking reproducible
        data_seed=42,
    )

    # 7. Initialize Trainer with both datasets
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # The trainer will now compute validation loss strictly from the test split
    )

    # Start execution
    trainer.train()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
