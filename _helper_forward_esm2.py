import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")

with app.setup:
    import os
    import gzip
    import shutil
    import urllib.request

    import marimo as mo
    import torch
    from transformers import EsmTokenizer, EsmForMaskedLM
    from Bio import SeqIO


@app.cell
def _():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (device,)


@app.cell
def _():
    # Download uniref50
    if not os.path.exists("data/uniref50.fasta.gz"):
        _url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniref/uniref50/uniref50.fasta.gz"
    
        print("Starting download of UniRef50 (approx. 7-9 GB compressed)...")
    
        _outdir = "data"
        if not os.path.exists(_outdir):
            os.makedirs(_outdir)
    
        with (
            urllib.request.urlopen(_url) as _response,
            open(os.path.join(_outdir, "uniref50.fasta.gz"), "wb") as _out_file,
        ):
            shutil.copyfileobj(_response, _out_file)
    
        print("\nDownload complete!")
    else:
        print("UniRef50 already exists. Skipping download.")
    return


@app.cell
def _():
    # Dummy protein sequence: first from UniRef50
    with gzip.open("data/uniref50.fasta.gz", "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            protein_sequence = str(record.seq)
            if len(protein_sequence) < 100:
                break  # Get only the first sequence with length < 100
    return (protein_sequence,)


@app.cell
def _(device):
    # Load ESM2 model
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)

    # Send to device in eval mode
    model.to(device)
    model.eval()
    return model, tokenizer


@app.cell
def _(device, model, protein_sequence, tokenizer):
    # Forward pass through the model
    with torch.inference_mode():
        _input = tokenizer(protein_sequence, return_tensors="pt").to(device)
        embedding = model(_input["input_ids"], output_hidden_states=True)
    return


if __name__ == "__main__":
    app.run()
