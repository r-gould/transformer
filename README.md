# Transformer

An implementation of a Transformer in PyTorch, with an application to English-German translation, as described in '[Attention Is All You Need](https://arxiv.org/abs/1706.03762)'.

src.transformer contains the stand-alone transformer model, and main.py demonstrates the model on a machine translation task on an English-German dataset Multi30k.

# Example

A trained model can then be used to translate a given sentence using the Translator class.

For example, a model trained for 15 epochs on Multi30k for English to German translation produces,

```
(see main.py for definitions)
>>> from src.translator import Translator
>>> translator = Translator(model, dataset)
>>> result = translator.translate(["a brown and white dog fetching a toy.",
                                   "two people standing next to a tree on the ground."],
                                   device=device)
[['ein', 'braun-weißer', 'hund', 'holt', 'ein', 'spielzeug', '.'],
 ['zwei', 'personen', 'stehen', 'neben', 'einem', 'baum', 'auf', 'dem', 'boden', '.']]
```

# References

**Attention Is All You Need:** https://arxiv.org/abs/1706.03762
