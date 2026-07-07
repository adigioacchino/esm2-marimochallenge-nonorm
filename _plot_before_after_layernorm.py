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
    from matplotlib.colors import LogNorm
    import torch
    from transformers import EsmTokenizer, EsmForMaskedLM
    from Bio import SeqIO

    import torch
    import torch.nn as nn


@app.cell
def _():
    mo.md("""
    # <center>  **Transformers without Normalization** </center>
    ## <center> Application to the ESM protein model </center>
    ---
    A wise man once said _"**Attention** is all you need!"_, but he probably forgot to add an important information:

    > <center> _"**Attention**, <ins>with proper normalization</ins>, is all you need!"_ </center>

    But what is Normalization? if you've ever bumped on a youtube video about LLM, You would already know that these powerful tool are composed as chains of arcane objects called _transformers_ which are able to recognize statistical interaction between distant token (aka letters/words etc.) in the input phrase.

    Transformers are truly amazing object, but they have one small Achilles' heel: they tend to
    """)
    return


@app.cell
def _():
    #UI objects
    choose_model = mo.ui.dropdown(
        options=["facebook/esm2_t30_150M_UR50D","facebook/esm2_t12_35M_UR50D", "facebook/esm2_t6_8M_UR50D"],
        value="facebook/esm2_t6_8M_UR50D",
    )

    choose_num_seqs = mo.ui.number(start=1, step=1, value=10)
    choose_len_down = mo.ui.number(start=1, step=1, value=100)
    choose_len_up = mo.ui.number(start=1, step=1, value=300)

    run_button = mo.ui.run_button(label = "Run Transformer")
    return (
        choose_len_down,
        choose_len_up,
        choose_model,
        choose_num_seqs,
        run_button,
    )


@app.cell
def _(
    choose_len_down,
    choose_len_up,
    choose_model,
    choose_num_seqs,
    run_button,
):
    mo.hstack(
        [
            mo.vstack([mo.md("Model selection"), choose_model]),
            mo.vstack([mo.md("Number of Sequences"), choose_num_seqs]),
            mo.vstack([mo.md("Min Length"), choose_len_down]),
            mo.vstack([mo.md("Max Length"), choose_len_up]),
            run_button
        ]
    )
    return


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
def _(
    choose_len_down,
    choose_len_up,
    choose_model,
    choose_num_seqs,
    device,
    run_button,
    uniref_ds,
):
    mo.stop(not run_button.value)

    # Example protein sequences from UniRef
    protein_sequences = sample_sequences(
        uniref_ds,
        num_seqs=choose_num_seqs.value,
        len_down=choose_len_down.value,
        len_up=choose_len_up.value,
    )

    # Load the ESM-2 model and tokenizer
    model_name = choose_model.value
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
    n_layers = int((len(activations) - 1)/2 + 1)
    choose_layer = mo.ui.slider(start=0, stop=n_layers-1, step=1, value=0, show_value=True, full_width=True)
    return (choose_layer,)


@app.cell
def _(choose_layer):
    mo.vstack([mo.md("Choose layer to plot"), choose_layer ])
    return


@app.cell
def _(choose_layer):
    selected_layer = int(choose_layer.value)
    return (selected_layer,)


@app.function
@mo.cache
def create_figure(activations, selected_layer):
    _n_layers = (len(activations) - 1)/2 + 1
    if selected_layer < _n_layers - 1:
        selected_keys = [_ for _ in activations.keys() if str(selected_layer) in _]
        n_cols = 2
        figsize = (5.5 * n_cols, 4.2 )
        fig, ax = plt.subplots(
            1, n_cols, figsize=figsize, constrained_layout=True
         )
        axes = ax.flatten()
    else:
        selected_keys = [_ for _ in activations.keys() if "emb_layer_norm_after" in _]
        n_cols=1
        figsize = (5.5 * n_cols, 4.2 )
        fig, ax = plt.subplots(
            1, n_cols, figsize=figsize, constrained_layout=True
         )
        axes = [ax]

    for _name, ax in zip(selected_keys, axes):
        before_tensors = activations[_name]["before"]
        after_tensors = activations[_name]["after"]

        # Concatenate all proteins' activation tensors (which may have different lengths)
        before_flat = torch.cat([t.flatten() for t in before_tensors]).numpy()
        after_flat = torch.cat([t.flatten() for t in after_tensors]).numpy()

        # Use hist2d with LogNorm for fast, high-density visualization
        h = ax.hist2d(
            before_flat,
            after_flat,
            bins=100,
            norm=LogNorm(),
            cmap="viridis"
        )

        # Add colorbar for each layer's heatmap
        fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)

        # Clean name for title to make it readable
        clean_name = (
            _name.replace("esm2.encoder.layer.", "Layer ")
            .replace(".attention.LayerNorm", " Attn LN")
            .replace(".LayerNorm", " LN")
        )
        ax.set_title(clean_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Before LayerNorm (Input)", fontsize=9)
        ax.set_ylabel("After LayerNorm (Output)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
    return fig


@app.cell
def _(activations, selected_layer):
    create_figure(activations, selected_layer)
    return


@app.cell
def _():
    # mo.accordion(
    #     items={
    #         f"Layer {_k}": create_figure(activations, _k)
    #         for _k in range(n_layers)
    #     },
    #     multiple=True,
    #     lazy=True,
    # )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
