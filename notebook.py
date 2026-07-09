import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")

with app.setup:
    import subprocess
    import sys
    import importlib
    import marimo as mo

    subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate>=1.1.0"])

    # Erase transformers from Python's active memory
    for module_name in list(sys.modules.keys()):
        if module_name.startswith("transformers") or module_name.startswith(
            "accelerate"
        ):
            del sys.modules[module_name]

    importlib.invalidate_caches()

    import os
    import copy
    import gzip
    import shutil
    import urllib.request
    import random
    from datasets import load_dataset
    import json
    import matplotlib.pyplot as plt
    import numpy as np

    import torch
    import torch.nn as nn
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
    from matplotlib.colors import LogNorm


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # <center>  **Transformers without Normalization** </center>
    ## <center> Application to the ESM protein model </center>
    ---
    A wise man once said _"**Attention** is all you need!"_, but he probably forgot to add an important piece of information:

    > <center> _"**Attention**, <ins>with proper normalization</ins>, is all you need!"_ </center>

    But what is Normalization? if you've ever come across a youtube video about LLM, you would already know that this powerful tool is composed of chains of arcane objects called _transformers_ which are able to recognize statistical interactions between distant tokens (aka letters/words etc.) in the input phrase.

    Transformers are truly amazing objects, but they have one small Achilles' heel: they often rescale the features. As a consequence, the data can collapse to only zeros, to a small subspace, or even explode in value. A solution to this is the Normalization layer, which rescales the output by centering it around the average, and normalizing it to the unit width. However, average and width change for every input/output pair, and therefore the parameters in this layer cannot be fixed - they are computed on the fly. This introduces a lot of complexity to the design of models, as choices have to be made about what procedure is used during the training, and deployment.

    In the work by [Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun and Zhuang Liu](https://arxiv.org/abs/2503.10622v2) they observe that actually in a large set of models, the Normalization layers learn to perform a very simple transformation. Even though each input is transformed linearly, when evaluated over many inputs (many training samples), a more complicated shape consistently appears - a logistic-like curve.

    In this notebook we made an arbitrary choice, and looked at a family of ESM protein models. Below you can check for yourself how Normalization layers in the model transform the features. You can choose between three models, and choose the number of tokens to average over. We observed that the complexity of the observed shape strongly depends on the length of the input protein sequence. You can explore this behavior by changing the `Min Length` and `Max Length` parameters. You might also observe that the complexity of shapes increases with the complexity of the model (**n** and **m** in '*esm2_t**n**_**m**M_UR50D*' count the number of transformer layers and the number of parameters).
    """)
    return


@app.cell
def _():
    choose_model = mo.ui.dropdown(
        options=[
            "facebook/esm2_t30_150M_UR50D",
            "facebook/esm2_t12_35M_UR50D",
            "facebook/esm2_t6_8M_UR50D",
        ],
        value="facebook/esm2_t6_8M_UR50D",
    )

    choose_num_seqs = mo.ui.number(start=1, step=1, value=10)
    choose_len_down = mo.ui.number(start=1, step=1, value=100)
    choose_len_up = mo.ui.number(start=1, step=1, value=300)

    run_button = mo.ui.run_button(label="Run Transformer")
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
            run_button,
        ]
    )
    return


@app.cell
def _():
    uniref_ds = load_dataset(
        "agemagician/uniref30", split="train", streaming=True
    ).shuffle(seed=1)
    return (uniref_ds,)


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
    model_name_first = choose_model.value
    tokenizer_first = EsmTokenizer.from_pretrained(model_name_first)
    model_first = EsmForMaskedLM.from_pretrained(model_name_first)

    # Send to device in eval mode
    model_first.to(device)
    model_first.eval()

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
    for _name, _module in model_first.named_modules():
        if isinstance(_module, nn.LayerNorm) and "lm_head" not in _name:
            _module.register_forward_hook(norm_hook(_name, activations))

    # Forward pass through the model
    with torch.inference_mode():
        for _protein_sequence in protein_sequences:
            _input = tokenizer_first(_protein_sequence, return_tensors="pt").to(device)
            model_first(_input["input_ids"])
    return (activations,)


@app.cell
def _(activations):
    n_layers = int((len(activations) - 1) / 2 + 1)
    choose_layer = mo.ui.slider(
        start=0,
        stop=n_layers - 1,
        step=1,
        value=0,
        show_value=True,
        full_width=True,
    )
    return (choose_layer,)


@app.cell
def _(choose_layer):
    mo.vstack([mo.md("Choose layer to plot"), choose_layer])
    return


@app.cell
def _(choose_layer):
    selected_layer = int(choose_layer.value)
    return (selected_layer,)


@app.cell
def _(activations, selected_layer):
    create_figure_first(activations, selected_layer)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We indeed see, that the shapes produced often end up forming a logistic function. The same observation in other models led the authors of the publication to propose a simpler alternative to the Normalization layer. Since the function seems to just reduce to simple $\tanh$, they propose to use

    $$
    DyT(\boldsymbol{x}) = \boldsymbol{\gamma} * \tanh(\alpha\boldsymbol{x})+\boldsymbol{\beta},
    $$

    where $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$ are vectors of weights with the dimension $|\boldsymbol{x}|$, and $\alpha$ a simple scalar. They refer to the layer as the DyT layer. Very surprisingly this layer seems to perform as well as a full normalization layer in many examples that they checked.

    ## <center> Training of ESM with DyT layer </center>
    ---

    In order to see this for ourselves, the notebook allows us to test this idea in the ESM protein model. Due to computational constraints we limit ourselves to a model with 8M parameters. Then we replace all the Normalization layers with the new $DyT$ layer, and retrain it from scratch. In order to make sure that our training is well done, we will also retrain the original model with standard LN normalization layer.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This computation is quite heavy, so we encourage the user to switch to the cuda kernel, offered by the **molab**. If the kernel was chosen successfully, the next line should print out: " *Device used: 'cuda'* "
    """)
    return


@app.cell
def _():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mo.callout(f"ℹ️ Device used for training: {device}", kind="info")
    return (device,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now that we checked what kernel we are using (we recommend cuda), we reload the original model, and initialize all the weights to random values. In the window below, you can inspect the structure of the model in detail.
    """)
    return


@app.cell
def _(device):
    model_name = "facebook/esm2_t6_8M_UR50D"  # this is a smaller model
    # model_name = "facebook/esm2_t12_35M_UR50D" # this is a larger model

    # the same tokenizer as before
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    # Now we load only the blueprint (skeleton) of the model, and put random weights
    config = EsmConfig.from_pretrained(model_name)
    model = EsmForMaskedLM(config)

    # Send to device in eval mode
    model.to(device)
    model.eval()
    return model, tokenizer


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In `pytorch` the layers can be easily replaced. First we define a new `DyT` layer class, and initialize the value of $\alpha_0=10.0$. The training is very sensitive to $\alpha_0$. Values like $\alpha_0=\{0.1, 0.5, 1.0\}$ significantly underperform the original model. The notebook allows you to play with different values by using the input window below. Once the `DyT` class is constructed, we define a function that goes through all the layers, and replaces all instances of `nn.LayerNorm` by `DyT`. The function makes sure that the dimension of features stays the same as in the original model.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```python
    class DyT(nn.Module):
        def __init__(self, num_features, alpha_init_value=10.0):
            super().__init__()
            self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

        def forward(self, x):
            x = torch.tanh(self.alpha * x)
            return x * self.weight + self.bias
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```python
    def replace_layernorm_with_dyt(module: nn.Module) -> None:
        for name, child in module.named_children():
            # If the child is a LayerNorm, replace it
            if isinstance(child, nn.LayerNorm):
                # ESM2 LayerNorms use a tuple for normalized_shape, e.g., (320,)
                # We extract the integer size to pass to your num_features
                num_features = child.normalized_shape[0]

                # Create your custom layer and swap it in
                custom_layer = DyT(num_features=num_features)
                setattr(module, name, custom_layer)
            else:
                # If it's not a LayerNorm, dig deeper into this child
                replace_layernorm_with_dyt(child)
    ```
    """)
    return


@app.cell
def _():
    parameter_alpha = mo.ui.number(
        start=0,
        stop=100,
        step=0.01,
        value=10.0,
        label="Set parameter $\\alpha$ for DyT layer",
    )
    return (parameter_alpha,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now we apply the function, and you can see that below all the Normalization layers are now replaced by DyT.
    """)
    return


@app.cell
def _(device, model, parameter_alpha):
    model_new = copy.deepcopy(model)
    replace_layernorm_with_dyt(model_new, alpha=parameter_alpha.value)
    model_new.to(device)
    model_new.eval()
    return (model_new,)


@app.cell
def _(tokenizer):
    max_length = 8 * 64  # Limit sequences to 100 amino acids for faster training
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now we can start the training. First, the original model, and then with the new model. The training loss along the training of the original model is already precomputed, and displayed below, for your convenience. **In order to start the training press the button.** Using the 'cuda' kernel offered by **molab** the training of each model should take about a minute. All the parameters, except for the learning rate were kept the same, while the learning rate equals to 1e-4 in the model with LN and 4e-4 in the model with DyT.
    """)
    return


@app.cell
def _():
    train_og_button = mo.ui.run_button(label="▶️ Train original model")
    return (train_og_button,)


@app.cell
def _(eval_dataset, model, tokenizer, train_dataset, train_og_button):
    mo.stop(not train_og_button.value)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # 6. Strict Hyperparameters for valid comparison
    training_args = TrainingArguments(
        output_dir="./esm2_comparison_run",
        max_steps=101,  # Fixed step limit ensures identical exposure to data
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        save_total_limit=2,
        eval_strategy="steps",  # Required for streaming iterable datasets
        eval_steps=100,  # Calculates validation loss every 1000 steps
        logging_steps=1,
        save_steps=100,
        gradient_checkpointing=False,
        learning_rate=1e-4,
        warmup_steps=10,
        max_grad_norm=1.0,
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        seed=42,  # Keeps the Data Collator's random masking reproducible
        data_seed=42,
        # torch_compile=True,
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
    train_new_button = mo.ui.run_button(label="▶️ Train the new model")
    return (train_new_button,)


@app.cell
def _(parameter_alpha, train_new_button, train_og_button):
    mo.vstack([train_og_button, train_new_button, parameter_alpha], justify="start")
    return


@app.cell
def _(eval_dataset, model_new, tokenizer, train_dataset, train_new_button):
    mo.stop(not train_new_button.value)
    data_collator_new = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # Strict Hyperparameters for valid comparison
    training_args_new = TrainingArguments(
        output_dir="./esm2_comparison_run_new",
        max_steps=101,  # Fixed step limit ensures identical exposure to data
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        save_total_limit=2,
        eval_strategy="steps",  # Required for streaming iterable datasets
        eval_steps=100,  # Calculates validation loss every 1000 steps
        logging_steps=1,
        save_steps=100,
        gradient_checkpointing=False,
        learning_rate=4e-4,
        warmup_steps=100,
        max_grad_norm=1.0,
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        seed=42,  # Keeps the Data Collator's random masking reproducible
        data_seed=42,
        # torch_compile=True,
    )

    # Initialize Trainer with both datasets
    trainer_new = Trainer(
        model=model_new,
        args=training_args_new,
        data_collator=data_collator_new,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # The trainer will now compute validation loss strictly from the test split
    )

    # Start execution
    trainer_new.train()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Visualization of the Evaluation and Training loss along the optimization. The precision reached is comparable to the precision of the original model, trained by Meta.
    """)
    return


@app.cell
def _(train_og_button):
    train_og_button
    state_file = (
        "./esm2_comparison_run/checkpoint-100/trainer_state.json"
        if os.path.exists("./esm2_comparison_run/checkpoint-100/trainer_state.json")
        else None
    )
    if state_file is not None:
        with open(state_file, "r") as f:
            _state_data = json.load(f)

        # Extract steps and loss values from the log history
        log_history = _state_data["log_history"]

        # Separate training loss and evaluation loss
        train_steps = [log["step"] for log in log_history if "loss" in log]
        train_loss = [log["loss"] for log in log_history if "loss" in log]

        eval_steps = [log["step"] for log in log_history if "eval_loss" in log]
        eval_loss = [log["eval_loss"] for log in log_history if "eval_loss" in log]
    return eval_loss, eval_steps, state_file, train_loss, train_steps


@app.cell
def _(train_new_button):
    train_new_button
    state_file_new = (
        ("./esm2_comparison_run_new/checkpoint-100/trainer_state.json")
        if os.path.exists("./esm2_comparison_run_new/checkpoint-100/trainer_state.json")
        else None
    )
    if state_file_new is not None:
        with open(state_file_new, "r") as f_new:
            state_data_new = json.load(f_new)

        # Extract steps and loss values from the log history
        log_history_new = state_data_new["log_history"]

        # Separate training loss and evaluation loss
        train_steps_new = [log["step"] for log in log_history_new if "loss" in log]
        train_loss_new = [log["loss"] for log in log_history_new if "loss" in log]

        eval_steps_new = [log["step"] for log in log_history_new if "eval_loss" in log]
        eval_loss_new = [
            log["eval_loss"] for log in log_history_new if "eval_loss" in log
        ]
    return (
        eval_loss_new,
        eval_steps_new,
        state_file_new,
        train_loss_new,
        train_steps_new,
    )


@app.cell
def _(
    eval_loss,
    eval_loss_new,
    eval_steps,
    eval_steps_new,
    parameter_alpha,
    state_file,
    state_file_new,
    train_loss,
    train_loss_new,
    train_loss_og_precomp,
    train_steps,
    train_steps_new,
    train_steps_og_precomp,
):
    og_prec = 2.44
    sizef = 14
    # Plot first pre-computed model values
    plt.plot(
        train_steps_og_precomp,
        train_loss_og_precomp,
        label="Training (LN)",
        color="blue",
        marker="o",
        alpha=0.1,
    )
    # plt.plot(
    #    eval_steps_og_precomp,
    #    eval_loss_og_precomp,
    #    label="Evaluation Loss (LN)",
    #    color="red",
    #    marker="x",
    # )

    # If the user trained their own original model plot that too
    if state_file is not None:
        plt.plot(
            train_steps,
            train_loss,
            label="Training (LN)",
            color="blue",
            marker="o",
        )
        plt.plot(
            eval_steps,
            eval_loss,
            label="Evaluation (LN)",
            color="black",
            marker="x",
        )
    # If the user trained their own new model plot it
    if state_file_new is not None:
        plt.plot(
            train_steps_new,
            train_loss_new,
            "--.",
            label="Training (DyT)",
            color="darkred",
        )
        plt.plot(
            eval_steps_new,
            eval_loss_new,
            "--x",
            label="Evaluation (DyT)",
            color="grey",
        )

    plt.plot([0, 100], [og_prec] * 2, "--", linewidth=1, color="black")
    plt.text(1, og_prec + 0.04, "OG precision", fontsize=sizef)

    plt.xlabel("Training Steps (DyT)", fontsize=sizef)
    plt.ylabel("Loss", fontsize=sizef)
    plt.title(
        f"Training and Evaluation Loss Comparison, $\\alpha =$ {parameter_alpha.value}",
        fontsize=sizef,
    )
    plt.legend(fontsize=sizef, frameon=False, loc='upper right')
    return


@app.function(hide_code=True)
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


@app.function(hide_code=True)
def create_figure_first(activations, selected_layer):
    _n_layers = (len(activations) - 1) / 2 + 1
    if selected_layer < _n_layers - 1:
        selected_keys = [_ for _ in activations.keys() if str(selected_layer) in _]
        n_cols = 2
        figsize = (5.5 * n_cols, 4.2)
        fig, ax = plt.subplots(1, n_cols, figsize=figsize, constrained_layout=True)
        axes = ax.flatten()
    else:
        selected_keys = [_ for _ in activations.keys() if "emb_layer_norm_after" in _]
        n_cols = 1
        figsize = (5.5 * n_cols, 4.2)
        fig, ax = plt.subplots(1, n_cols, figsize=figsize, constrained_layout=True)
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
            cmap="viridis",
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


@app.class_definition(hide_code=True)
class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=10.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


@app.function(hide_code=True)
def replace_layernorm_with_dyt(module: nn.Module, alpha: float = 10.0) -> None:
    """
    Recursively searches a PyTorch model for nn.LayerNorm modules
    and replaces them with the custom DyT layer.
    """
    for name, child in module.named_children():
        # If the child is a LayerNorm, replace it
        if isinstance(child, nn.LayerNorm):
            # ESM2 LayerNorms use a tuple for normalized_shape, e.g., (320,)
            # We extract the integer size to pass to your num_features
            num_features = child.normalized_shape[0]

            # Create your custom layer and swap it in
            custom_layer = DyT(num_features=num_features, alpha_init_value=alpha)
            setattr(module, name, custom_layer)
        else:
            # If it's not a LayerNorm, dig deeper into this child
            replace_layernorm_with_dyt(child, alpha=alpha)


@app.class_definition(hide_code=True)
class StreamingUniRefDataset(IterableDataset):
    def __init__(self, tokenizer, split="train", max_length=1024, seed=1):
        self.tokenizer = tokenizer
        self.split: str = split
        self.max_length: int = max_length
        self.seed: int = seed
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


@app.cell(hide_code=True)
def _():
    train_loss_og_precomp = [
        3.4790120124816895,
        3.4713423252105713,
        3.3997716903686523,
        3.281043767929077,
        3.191840171813965,
        3.149264097213745,
        3.0645675659179688,
        3.0586206912994385,
        3.0050652027130127,
        2.996073007583618,
        2.999319076538086,
        2.9976134300231934,
        3.0016891956329346,
        2.9463517665863037,
        2.954683303833008,
        2.941167116165161,
        2.9394967555999756,
        2.943321704864502,
        2.916232109069824,
        2.9267561435699463,
        2.898732900619507,
        2.893810987472534,
        2.861203193664551,
        2.884669780731201,
        2.8641276359558105,
        2.8433263301849365,
        2.8692564964294434,
        2.876488447189331,
        2.8567521572113037,
        2.8488802909851074,
        2.8181281089782715,
        2.785329818725586,
        2.816866636276245,
        2.804933547973633,
        2.8360254764556885,
        2.792471170425415,
        2.8073348999023438,
        2.7680583000183105,
        2.768831729888916,
        2.7843246459960938,
        2.7445991039276123,
        2.7720043659210205,
        2.708559989929199,
        2.8043060302734375,
        2.7385740280151367,
        2.725388526916504,
        2.7335050106048584,
        2.7093684673309326,
        2.7321765422821045,
        2.75209379196167,
        2.717442750930786,
        2.7273409366607666,
        2.7476418018341064,
        2.6811773777008057,
        2.745795249938965,
        2.7245304584503174,
        2.7490625381469727,
        2.7078254222869873,
        2.7115366458892822,
        2.70440411567688,
        2.709927558898926,
        2.7460572719573975,
        2.6954030990600586,
        2.7279374599456787,
        2.6930084228515625,
        2.698761224746704,
        2.739962339401245,
        2.7155139446258545,
        2.728412628173828,
        2.7035579681396484,
        2.7126238346099854,
        2.670447587966919,
        2.726806163787842,
        2.6547999382019043,
        2.674043655395508,
        2.696397304534912,
        2.6888720989227295,
        2.6871654987335205,
        2.6838223934173584,
        2.7047741413116455,
        2.6851141452789307,
        2.7064545154571533,
        2.6724605560302734,
        2.669290781021118,
        2.717341184616089,
        2.6621532440185547,
        2.696638822555542,
        2.6997692584991455,
        2.6998331546783447,
        2.6618871688842773,
        2.7069151401519775,
        2.6942381858825684,
        2.715121269226074,
        2.6699461936950684,
        2.7371037006378174,
        2.6793854236602783,
        2.6477065086364746,
        2.711721181869507,
        2.6924729347229004,
        2.670194149017334,
    ]

    train_steps_og_precomp = np.arange(1, len(train_loss_og_precomp) + 1)

    eval_loss_og_precomp = [2.6847891807556152]

    eval_steps_og_precomp = [1]
    return train_loss_og_precomp, train_steps_og_precomp


if __name__ == "__main__":
    app.run()