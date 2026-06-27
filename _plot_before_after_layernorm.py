import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")

with app.setup:
    import os
    import gzip
    import shutil
    import urllib.request

    from datasets import load_dataset
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    from transformers import EsmTokenizer, EsmForMaskedLM
    from Bio import SeqIO

    import torch
    import torch.nn as nn


@app.cell
def _():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (device,)


@app.cell
def _():
    # Get uniref50 from HF in streaming mode
    uniref_ds = load_dataset(
        "agemagician/uniref30", split="train", streaming=True
    ).shuffle(seed=1)
    return (uniref_ds,)


@app.function
def sample_sequences(dataset, num_seqs: int, len_down: int, len_up: int):
    results = []
    for x in dataset:
        seq = x["text"]
        L = len(seq)
        if L >= len_down and L <= len_up:
            results.append(seq)
        if len(results) >= num_seqs:
            break
    return results


@app.cell
def _(uniref_ds):
    # Example protein sequences from UniRef
    protein_sequences = sample_sequences(uniref_ds, num_seqs=5, len_down=100, len_up=200)
    return (protein_sequences,)


@app.cell
def _(device, protein_sequences):
    # Load the ESM-2 model and tokenizer
    model_name = "facebook/esm2_t12_35M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)

    # Send to device in eval mode
    model.to(device)
    model.eval()

    # Create a dictionary to store pre- and post-norm activations
    def norm_hook(layer_name, activation_dict):
        def hook(
            module,
            inputs,
            output,
        ):
            if layer_name in activation_dict:
                _before = activation_dict[layer_name]["before"]
                _after = activation_dict[layer_name]["after"]
                activation_dict[layer_name] = {
                    "before": _before + [inputs[0].detach().cpu()],
                    "after": _after + [output.detach().cpu()],
                }
            else:
                activation_dict[layer_name] = {
                    "before": [inputs[0].detach().cpu()],
                    "after": [output.detach().cpu()],
                }

        return hook

    activations = {}
    for _name, _module in model.named_modules():
        if isinstance(_module, nn.LayerNorm) and "lm_head" not in _name:
            _module.register_forward_hook(norm_hook(_name, activations))

    # Forward pass through the model
    with torch.inference_mode():
        for _protein_sequence in protein_sequences:
            _input = tokenizer(_protein_sequence, return_tensors="pt").to(
                device
            )
            model(_input["input_ids"])
    return (activations,)


@app.cell
def _(activations):
    n_layers = len(activations)
    n_cols = 3
    n_rows = (n_layers + n_cols - 1) // n_cols
    figsize = (5.1 * n_cols, 4.0 * n_rows)
    fig, ax = plt.subplots(
        n_rows, n_cols, figsize=figsize, constrained_layout=True
    )
    axes = ax.flatten()

    for i, _name in enumerate(activations.keys()):
        before_tensors = activations[_name]["before"]
        after_tensors = activations[_name]["after"]

        # Concatenate all proteins' activation tensors (which may have different lengths)
        before_flat = torch.cat([t.flatten() for t in before_tensors])
        after_flat = torch.cat([t.flatten() for t in after_tensors])

        curr_ax = axes[i]
        curr_ax.scatter(
            before_flat, after_flat, alpha=0.3, s=2, color="indigo"
        )

        # Clean name for title to make it readable
        clean_name = (
            _name.replace("esm2.encoder.layer.", "Layer ")
            .replace(".attention.LayerNorm", " Attn LN")
            .replace(".LayerNorm", " LN")
        )
        curr_ax.set_title(clean_name, fontsize=11, fontweight="bold")
        curr_ax.set_xlabel("Before LayerNorm (Input)", fontsize=9)
        curr_ax.set_ylabel("After LayerNorm (Output)", fontsize=9)
        curr_ax.grid(True, linestyle="--", alpha=0.5)

    # Remove any unused subplot slots
    for j in range(n_layers, len(axes)):
        fig.delaxes(axes[j])

    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
