import torch

from torch.nn.utils.rnn import pad_sequence

class Translator:
    
    def __init__(self, model, dataset):
        
        self.model = model
        self.dataset = dataset

    def translate(self, src_strs, device="cuda"):

        src_field, trg_field = self.dataset.src, self.dataset.trg
        src_vocab, trg_vocab = src_field.vocab, trg_field.vocab
        unk_idx, pad_idx = src_vocab["<unk>"], src_vocab["<pad>"]
        sos_idx, eos_idx = trg_vocab["<sos>"], trg_vocab["<eos>"]

        def src_str_to_id(token):
            id = src_vocab.stoi.get(token)
            if id is None:
                id = unk_idx
            return id

        def trg_id_to_str(id):
            return trg_vocab.itos[id]

        src_tokens = list(map(src_field.tokenize, src_strs))
        src_idxs = [torch.tensor(list(map(src_str_to_id, tokens)), dtype=torch.int64)
                    for tokens in src_tokens]
        src_idxs = pad_sequence(src_idxs, batch_first=True, padding_value=pad_idx).to(device)

        trg_ids = self.generate_trg_idxs(src_idxs, sos_idx, eos_idx)
        trg_tokens = [list(map(trg_id_to_str, ids)) for ids in trg_ids]
        return trg_tokens

    @torch.no_grad()
    def generate_trg_idxs(self, src, sos_idx, eos_idx, stopping_len=64):

        batch_size, _ = src.shape
        trg = sos_idx * torch.ones(batch_size, 1, dtype=torch.int64).to(src.device)

        idxs = [i for i in range(batch_size)]
        trg_idxs = [None for _ in range(batch_size)]
        len_counter = 0

        while True:
            logits = self.model(src, trg)
            pred_idxs = torch.argmax(logits, dim=-1)
            next_idxs = pred_idxs[:, -1].unsqueeze(-1)
            trg = torch.cat((trg, next_idxs), dim=-1)

            eos_mask = (next_idxs == eos_idx).squeeze(-1)

            for i, stop in reversed(list(enumerate(eos_mask))):
                if not stop:
                    continue
                idx = idxs[i]
                trg_idxs[idx] = trg[i, 1:-1].tolist()
                del idxs[i]

            src = src[eos_mask == 0]
            trg = trg[eos_mask == 0]

            if len(src) == 0:
                break

            len_counter += 1
            if len_counter >= stopping_len:
                for i in range(len(trg)):
                    idx = idxs[i]
                    trg_idxs[idx] = trg[i, 1:].tolist()
                break

        return trg_idxs