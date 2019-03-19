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

Actually, in each sub-layer in each encoder (same to decoder), it has a residual connection around it and followed by a layer-normalization step.

## End to end

The input word will be turned into a vector via word embedding algorithm. The embedding only happens in the bottom-most (first) encoder. Each encoder has the common abstraction with the same size of input (say 512). The input for the bottom encoder will be the word embedding, while the others will take the output of previous stack encoder as the input.

Each word flows through its own path in the encoder. The dependencies between these paths only in the self-attention layer while the feed-forward layer does not. Then they could be executed in parallel in the feed-forward layer. 

# Self-attention in Detail

There are six steps to calcualte the self-attention:
- Step 1: Create three vectors from each of the encoder's input (word) vectors: *Query vector, Key vector, Value vector*.
  - What are they?
    - Smaller in dimension (64) than the embedding vector (512). However, no need to be samller, but this is architecture choise
  - How to calculate that three vectors?
    - Trained during the training process
  - Why we need these three vectors? 
- Step 2: Calculate a score
  - What?
    - Score each word of the input sentence against the current word
    - The score determines how much focus to place on other parts of input sentence when encoding a centain position
  - How?
    - Take dot product of `query vector` with the `key vector` of the respective word we're scoring, like q1 * k1, q1 * k2 etc.
- Step 3 & Step 4: divide score by 8 (i.e. √d), then pass through a softmax operation to normalize the score.
  - The softmax score determines how much each word will be expressed at this position
- Step 5: multiply each value vector by the softmax score
- Step 6: sum up the weighted value vectors
  - The result is the output of self-attention layer for current position

*Self-attention calculation is done via matrix.*

## Many heads approach
*Why we need many heads method?*

It improves the performance of the attention layer in two ways:
- Expand the model's ability to focus on different positions
- Give the attention layer multiple "representation subspaces": multi-headed attention multiple sets of Query/Key/Value weight matrices. Each of the sets is randomly initalized which means projecting embeddings into different representation subspace

*How to connect to the feed forward neural network?*

The feed-forward network expects only one matrix (one vector for each word) instead of multiple. To convert it to one simple, we concat the matrices then multiple them by an additional weights matrix. *The matrix is trained jointly with the model*.

## Positional encoding

To describe the order info of words, the transformer adds a vector to each input embedding. 

*How to generate the vector?*
> TODO: add the detail here

# Decoder

The self attention layers slightly different from that in the encoder. The attention layer only allowed to earlier positions in the output sequence by masking future positions before the softmax step.

*Encoder-Decoder Attention* layer works like multi-head self-attention, except it creates its Query matrix from layer below it and take Keys and Values matrix from output of the encoder stack. 


## References
- [Illustrated transformer](https://jalammar.github.io/illustrated-transformer/)