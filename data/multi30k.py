import spacy

from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets import Multi30k

class Multi30kDataset:

    def __init__(self, exts=(".en", ".de")):

        en = spacy.load("en_core_web_sm")
        de = spacy.load("de_core_news_sm")

        en_tokenizer = lambda text: [token.text for token in en.tokenizer(text)]
        de_tokenizer = lambda text: [token.text for token in de.tokenizer(text)]

        src_tokenizer = en_tokenizer if exts == (".en", ".de") else de_tokenizer
        trg_tokenizer = de_tokenizer if exts == (".en", ".de") else en_tokenizer

        self.src = Field(tokenize=src_tokenizer, init_token="<sos>", 
                            eos_token="<eos>", lower=True, batch_first=True)
        
        self.trg = Field(tokenize=trg_tokenizer, init_token="<sos>", 
                            eos_token="<eos>", lower=True, batch_first=True)
        
        self.dataset = Multi30k.splits(exts=exts, fields=(self.src, self.trg))
        self.vocab_built = False

    def dataloaders(self, batch_size, min_freq):

        train, valid, test = self.dataset

        self.src.build_vocab(train, min_freq=min_freq)
        self.trg.build_vocab(train, min_freq=min_freq)
        self.vocab_built = True

        return BucketIterator.splits((train, valid, test), batch_size=batch_size)

    def vocab_sizes(self):
        
        assert self.vocab_built
        return len(self.src.vocab), len(self.trg.vocab)
        


    

