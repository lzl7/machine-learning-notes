# High level components

Encoding component is a stack of encoders (six of them and on top of each other). Decoding component is a stack of decoders with the same number.

## Encoder
Each encoder is identical in structure (not shared weights), and contains two sub-layers:
- *Self-attention*, 'global view' looks other words as well when encoding a specific word
- *Feed forward neural network*, takes the outputs from self-attention and independently apply to each position.

## Decoder
Similar to encoder and has those two layers, in additional there is an attention layer in the middle.
- *Self-attention*
- *Encoder-decoder attention*, focus on relevant part of the input sentence
- *Feed forward neural network*

## End to end

The input word will be turned into a vector via word embedding algorithm. The embedding only happens in the bottom-most (first) encoder. Each encoder has the common abstraction with the same size of input (say 512). The input for the bottom encoder will be the word embedding, while the others will take the output of previous stack encoder as the input.

Each word flows through its own path in the encoder. The dependencies between these paths only in the self-attention layer while the feed-forward layer does not. Then they could be executed in parallel in the feed-forward layer. 


## References
- [Illustrated transformer](https://jalammar.github.io/illustrated-transformer/)