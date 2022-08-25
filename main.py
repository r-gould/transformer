import yaml
import torch
import torch.nn as nn

from src.transformer import Transformer
from src.scheduler import Scheduler
from src.trainer import Trainer
from data.multi30k import Multi30kDataset
from src.translator import Translator
from utils import plot_stats

def main(dataloaders, params, epochs, warmup_steps, 
         vocab_in, vocab_out, max_seq_len, pad_idx, 
         load_model=False, save_model=True, device="cuda"):

    model = Transformer(**params, vocab_in=vocab_in, vocab_out=vocab_out, 
                        max_seq_len=max_seq_len, pad_idx=pad_idx).to(device)

    if load_model:
        model.load_state_dict(torch.load("saved/transformer.pt"))

    loss = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optim = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=10e-9)
    scheduler = Scheduler(optim, params.get("d_model"), warmup_steps)

    trainer = Trainer(dataloaders, loss, scheduler)
    train_losses, valid_losses = trainer.train(model, epochs, validate=True, 
                                               save_model=save_model, 
                                               device=device)

    plot_stats(train_losses, valid_losses)
    return model

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    batch_size = 128
    dataset = Multi30kDataset(exts=(".en", ".de"))
    dataloaders = dataset.dataloaders(batch_size, min_freq=2)

    with open("transformer/params.yaml", "r") as stream:
        params = yaml.safe_load(stream)

    epochs = 50
    warmup_steps = 4000
    vocab_in, vocab_out = dataset.vocab_sizes()
    max_seq_len = 128
    pad_idx = 1

    model = main(dataloaders, params, epochs, warmup_steps, 
                 vocab_in, vocab_out, max_seq_len, pad_idx, 
                 save_model=True, device=device)

    # Example

    translator = Translator(model, dataset)
    result = translator.translate(["a brown and white dog fetching a toy.",
                                   "two people standing next to a tree on the ground."],
                                   device=device)
    print("Translation:", result)
