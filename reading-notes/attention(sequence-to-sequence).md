# Sequence to Sequence Model

A sequence to sequence model takes a sequences of items and outputs another sequence of items.

In underlying, the model composes of an `encoder` and `decoder`.
- **Encoder**
  - *what*: tends to be recurretn neural network (RNN)
  - *how*: it processes each input and compiles the information into a vector (i.e. `context`)
- **Decoder**
  - *what*: tends to be recurrent neural network (RNN)
  - *how*: it takes the `context` from encoder, and produces the output sequence item by item
  - *When*: after `encoder` processing entire input sequence 

The `context` vector's size can be changed during setting up the model. It usually be 256, 512, 0r 1024.

RNN usually takes two inputs (vectors): input element and hidden state, and output new (hidden) state. The encoder updates the hidden state based on the previous state and the current input, and the decoder also maintains a hidden states that it passes from one step to the next.

## Attention

Problems in above encoder-decoder sequence to sequence model:
- Context vector will be the bottleneck, not good to long sentences

Attension model different from classic seq2seq model:
- Encoder passes all hidden states to decoder, instead of only the last hidden state
  - A lot more data, how much does it impact the performance?
- Decoder does an extra step before producing the output
  - How it works?
    - Look at a set of encoder hidden states
    - Give each hidden state a score (TODO: how????)
    - Multiply each hidden state by its softmaxed score. The softmaxed score is used to amplify or drown out the hidden state. And the sum up vector will be the context.
  - What is the steps?
    - Decoder takes the embedding of `<END>` and an initial decoder hidden state
    - Decoder processes inputs and generate the new hidden state (*h*) and a output. The output will be discarded and replaced by the last step output bellow
    - Attention step: use the *encoder hidden states* and *h* vector to calculate the context (*c*)
    - Concate h and c into one vector
    - Pass that vector from above step to a `feedforward neural network`
    - The *output* of feedforward neural network as the output of this time step
    - Repeat above for next time steps 

The model actually learned from the training phase how the language pair aligns the words.

## References
- [Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [Illustrated transformer](http://jalammar.github.io/illustrated-transformer/)
- [Harvard attention paper](http://nlp.seas.harvard.edu/2018/04/03/attention.html)