# Pretrain Model

Pretrain model is quite popular in image/video processing, like ImageNet. The pretrain method basically use large dataset to train the basic generical model via deep learning technology. And later in different machine learning tasks, they could start based on the pretrain model and continue train the specific task model for various tasks.

*Why we need pretrain model?*

There are several advantages by leveraging the pretrain model:
- Fasten the training process.
- Good for small dataset. Say in specific domains it might not have enough dataset
- Better for optimization. The trained model could provide better initialization parameters for the training model
- Generalization. Usually, the pretrain model is trained with large various dataset, and it could provide better generalization model. Later, when training the new model, especiall with small dataset, the generalization is very helpful.

*Why pretrain model works?*

The pretrain model has very good capability to extract low level features and comes with generalization which does not specifically for certain concrete tasks. Later in the concrete task training, it could leverage the pretrain model to extract the low leverl features and focus on optimizing the high level specific task.

*How to use the pretrain models?*

There are two ways to blend the pretrain model in the model training process:
1. **Frozon**. THe shadow layer's parameters do not change (leverage what it has from the pretrain model), and start from there to train the model for the task. 
2. **Fine-Tuning**. Tune the parameters from the pretrain model for the specific task, and make it better fit for the new task.

*Does the new training need to have the same network structure with the pretrain model?*

Yes, for the low/shadow layers or the starting layers.

# Pretrain Model in NLP

## Word Enbedding

[Language Model](https://en.wikipedia.org/wiki/Language_model) is the probability distribution over sequences of words, i.e. to quantify the quality of a sentence. There are several kinds of models:
- **Unigram**, each word is independent
- **N-Gram**, the word depends on the other (n-1) words
- **Neural Network**, nerual language models (i.e. continuous space language models), use continuous of representations or embeddings of words

### Nerual Network Language Model (NNLM)

It use the nerual network architecture to model the language and base on that to learn the language model. The training for NNLM is: given a couple of words and predict the next word.

After the model training, the network could be able to predict the output for given sequence words. During the learning, it also generates a matrix (row size is the dictionary size, each row the embedding of each word) which projects the input word (with one-hot representation) into a vector.

The popular tools to build the word embedding is: Word2Vec and Glove.

### Word2Vec

The architecture of Word2Vec are similar to NNLM, but the training method is different.

Word2Vec uses two training methods:
- CBOW
  - Mask a word from a sentence, and leverage the context (Context-before and Context-After) to predict the masked word
- Skip-gram
  - Opposite to CBOW: given a word and predict the context words.

Btw, the NNLM only use the context-before as input and predict the word.

*What is the disadvantages?*

- Not able to distinguish the same word, different semantics.

*Implement tech*

- Negative Sampling:
  - Instead of finding the most posibility word from the whole dictionary list, for a given word randomly sample other noise words

- Hierarchical Softmax
  - Dramatically reduce the complexity from O(n) to O(logn)
  - It uses the binary tree, where leaves represent probabilities of words

### ELMO (Embedding from language Models)

The core difference in ELMO is about deep contextualized: deep and context. The word embedding previous is static, after training the word representation is frozon and will NOT change in different contexts.

In ELMO, it uses the language model to learn a good word embedding for words (not semantic info), however when using the learned word enbedding, it will adjust the representation of the word embedding.

*How ELMO dynamic adjust the word embedding?*

ELMO contains two phases:
- Language model training
  - Dual Layers Bi-LSTM
    - Benefits:
      - Bottom layer: word embedding
      - Middle layer: syntax info embedding
      - Top layer: semantic info embedding
- Take the trained new features from all network layers from Word Embedding and used in the downstream task.
  - Each word in ELMO has three embedding, and each embedding accumulated with cooresponding weight as a new embedding vector
  - The new embedding vector will be a new feature for the downstream task training

*Comparing to other similar methods*

**TagLM** uses the similar idea to do NER.

**ULMFiT** ([Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)) uses three phases.

*Disadvantages*

- Feature extractions weak, comparing with Transformer

### GPT (Generative Pre-Training)

GPT uses two phases as well:
- Pretrain with language model
- Fine-tuning to process downstream task
 
Differnces (vs ELMO):
- Feature extraction via Transformer instead of RNN
- Still language model as the task, but use the one-direction language model
  - Use the context-before to predict the word
  
*How to use GPT?*

The downstream task network has to be the same as the GPT, take the pretrained GPT parameters can directly used and then, use the current task to train the network parameter (fine-tuning) to adapt to resolve the current task. The same as the approach in image domain.


### BERT

Bert use the same two-phase model as GPT, but the language model is using the bi-language model with larger dataset.

When modifying the downstream task structure, BERT uses different solutions.

*How BERT construct bidi-language model?*

- Masked language model
  - Essence same as CBOW
    - 15% words randomly selected and masked by `[Mask]`. In order to void the overfit (learn) to detect `[Mask]` which invisuable for prediction
      - 80% marked as [Mask]
      - 10% replaced as another word
      - 10% no change
- Next Sentence Prediction
  - The corpus are corrected two sequences
  - The second sentence is reordered as the first sentence

Then the training task is not only the Masked language model but also the sentence relationship prediction. Its pretrain is a multiple tasks.

Input of BERT:
- Position info embedding
- Word embedding
- Sentence embedding

*What affect BERT quality?*

- Major factor: Bi-direction language model

## References
- [从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史](https://zhuanlan.zhihu.com/p/49271699)