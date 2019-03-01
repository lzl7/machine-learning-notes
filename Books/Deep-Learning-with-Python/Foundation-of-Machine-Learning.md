# Foundation of Machine Learning

## Four Branches of Machine Learning

### 1. Supervised learning

Example problems: 
- Binary classification
- Multiclass classification
- Scalar regression
- Sequence generation
  - Like given a picture, predict caption describing it
  - It can sometimes be reformulated as a series of classification problems (such as repeatedly predicting a word or token in a sequence)
- Syntax tree prediction
  - Given a sentence, predict its decomposition into a syntax tree
- Object detection
  - Given a picture, draw a bounding box around certain objects inside the picture. It can also be expressed as a classification problem, or as a joint classification and regression problem
- Image segmentation
  - Given a picture, draw a pixel-level mask on a specific object

Basically, the supervised learning is to learn to map input data to known targets (also called annotations) by given a set of examples.

### 2. Unsupervised learning
*Dimensionality reduction* and *clustering* are well-known categories of unsupervised learning.

### 3. Self-supervised learning
Sort of supervised learning without humans involved (no human-annotated labels), the labels are generated from the input data, typically using a heuristic algorithm.

Examples:
- Autoencoders, the targets are the input
- Frame prediction in video

### 4. Reinforcement learning
The agent receives information about its enironment and learns to choose actions that will maximize some reward.

#### Classification and regression glossary
- Sample or input -- one data point that used to train the model
- Prediction or ouput -- the outcome from the model
- Target -- the truth, what the ideally prediction for the model
- Prediction error or loss value -- A measure of distance between your model's prediction and the target
- Classes -- A set of possible labels to choose from in a classification problem
- Label -- A specific instance of a class annotation in a classification probelm
- Ground-truth or annotations -- All targets for a dataset, typically collected by humans
- Binary classification -- A classification task where each input sample should be categorized into two exclusive categories
- Multiclass classification -- A classification task where each input sample should be categorized into more than two, like handwritten digits
- Multilabel classification -- A classification task where each input sample can be assigned multiple labels, like an image contains cat and dog
- Scalar regression -- A task where the target is a continuous scalar value, house prices prediction
- Vector regression -- A task where the target is a set of continuous scalar value
- Mini-batch or batch -- A small set of samples (typically between 8 and 128) that are processed simultaneously by the model

## Evaluate machine learning models

*What is a good model?*

The model generalizes enough, which could perform well on never-before-seen data.

*Dataset categories in training*
- Training set, the dataset used for training the model
- Validation set, the dataset used for evaluating your model on validation data
- Test set, the dataset for testing the generalization quality of the model

*Why have validation set and test set?*

Validation set is useful to tune the training parameters/hyper-parameters and see the quality for the mode during the training. Test set (never-before-seen dataset) will help to detect whether these is overfitting for the trained model.

*Information leaks*, means every time you tune a hyperparameter of the model based on the performance on the validation set, some info about the validation data leaks into the model.

*How to split the dataset?*

Three classic methods (when data is little):
- Simple hold-out validation -- simply split the (exclude test set) data into training set and hold-out validation set, and iteratively train on training set, validate on validation set
- K-fold validation -- split the data into K partitions of equals size. For each partition, train the a model with the rest K-1 partitions, the average of K scores as the final one.
- Iterated K-fold validation with shuffling -- by applying K-fold validation mutiple time, and shuffling the data every time before splitting the data. Scenarios: have relatively little data, like Kaggle competitions. However, it is expensive since P(iteration) * K models are trained and evaluated.

*Some tips for training model*
- Data representativeness -- Both the training set and test set should be representative of hte data at hand. Usually, we should randomly shuffle the data before splitting it into training and test sets.
- The arrow of time -- If the data is in sequence, should not randomly shuffle the data before splitting it, like stock data. Other wise, there is a temporal leak: use the future data for training. We should make sure the test set data is posterior to the training set data
- Redundancy in the data -- make sure training set and validation set are disjoint. If the redundancy data in both training and validation sets, it will impact the model.

## Data preprocessing, feature engineering and feature learning

*What is data processing?*

Data processing aims at making the raw data more amenable to amenable to neural networks:
- Vectoraization -- we need to turn the input data (sound, images, text etc.) into tensors, called data vectorization.
- Normalization -- Like normalize the value range into [0, 1] which makes the network learning more easier. However, we don't need to have each feature with mean of 0 and standard deviation of 1.
  - Take small values -- typically, most values should be in [0, 1]
  - Be homogenous -- All features should take values in roughly the same range
- Handling missing values -- Generally, it's safe to input missing values as 0 in neural networks. The network will learn from exposure to the data that the value 0 means missing data and will start ignoring the data.
- Feature extraction

*What is feature engineering?*

Feature engineering is the process of using the domain knowledge and create the derived data from the original data. The essence of feature engineering: making a problem easier by experssing it in a simpler way. 

Don't we need to worry about feature engineering in deep neural networks that could automatically extracting useful features from raw data? No.
- Good features allow you solve problems more elegantly while using fewer resources
- Good features let you solve a problem with far less data

## Overfitting and underfitting

*Optimization* refers to hte process of adjusting a model to get the best possible performance on the training data, while *generalization* referes to how well the trained model performs on data it has never seen before.

*Underfit*: the network hasn't yet modeled all relevant patterns in the training data. 

*Overfit*: the model performs well in the training/evaluation set, but worse in test set (new data). The best solution is to get more training data, and also there are other regularization solutions.

### Tackling overfitting

*1. Reducing the network's size*

Reduce the model size (i.e. the number of learnable parameters) is the simplest way to prevent overfitting. In deep learning, the number of learnable parameters in a model is often referred to as the model's capacity.

*2. Adding weight regularization*

- L1 regularization -- the cost added is proportional to the absolute value of the weight coefficients (the L1 norm of the weights)
- L2 regularization -- the cost added is proportional to the square of the vlaue of the weight coefficients (the l2 norm of the weights). L2 is also called weight decay in the context of neural networks.

*3. Adding dropout*

*Dropout* is one of the most effective and most commonly used regularization techniques for neural networks. It randomly drops out a number of output features of layer during training, usually set to [0.2, 0.5].

## The workflow of machine learning

*1. Defining the problem and assembling a dataset*

The first step should be konwing what your inputs and outputs are, and what data you will use.
- What will be the input data? What are you trying to predict?
- What type of problem are you facing? Binary classification? Multiclass classification? Scalar regression? Vector regression? Multiclass, multilabel classification?

Be aware of the hypotheses:
- The hypothesize that the outputs can be predicted given the input
- The hypothesize that the available data is sufficiently informative to learn the relationship between inputs and outputs

There is another probelm should be aware: *nonstationary problems*. The data changed time by time, like the clothing recommendation.

Machine learning can only used to memorize patterns that presents in the training data.

*2. Choosing a measure of success*

To achieve success, we need to define what the success means: accuracy? Precision and recall? Customer-retention rate?

The metrics guides the choice of a loss function: what model will optimize.

Tips:
- For balanced-classification problems, accuracy and *area under the receiver optating characteristic curve* (ROC AUC).
- For imbalanced-classification problems: precision and recall
- For ranking problems or multilabel classification: mean average precision

https://kaggle.com shows a wide range of problems and evaluation metrics.

*3. Deciding on an evaluation protocol*

There are three common evalution protocols:
- Maintaining a hold-out validation set -- when have planty of data
- Doing K-fold cross-validation -- have too few samples for hold-out validation to be reliable
- Doing iterated K-fold validation -- for performing highly accurate model evaluation when little data is available

*4. Preparing your data*

Previous steps: know what the training on, what optimize for, how to evaluate the approach. Right now need to prepare the data:
- Data should be formatted as tensors
- Value in the tensor usually be normalized (scaled) to small values, like [0, 1]
- If different features take values in different ranges (heterogeneous data), then the data should be normalized
- Might do some feature engineering, especially for small-data problems

*5. Developing a model that does better than a baseline*

Three key choices to build first working model:
- Last-layer activation -- this establishes useful constrains on the network's output.
- Loss function -- this should match the type of problem to solve.
- Optimization configuration -- what optimizer will be used? What will be the learning rate? In most case, it's safe to go with RMSProp and its default learning rate.

Suggestion on last-layer activation function and a loss function:

Problem type | Last-layer activation | Loss function
--- | --- | ---
Binary classification | sigmoid | binary_crossentropy
Multiclass, single-label classification | softmax | categorical_crossentropy
Multiclass, multilabel classification | sigmoid | binary_crossentropy
Regression to arbitrary values | None | mse
Regression to values between 0 and 1 | sigmoid | mse or binary_crossentropy


*6. Scaling up: developing a model that overfits*

When training model, how to know the model is sufficiently powerful? Enough layers and parameters?

Develop an overfit model could help to figure out how big a model you’ll need by:
- Add layers (deeper)
- Make the layers bigger (wider) 
- Train for more epochs


*7. Regularizing your model and tuning your hyperparameters*

Regularization solutions:
- Add dropout
- Try different architecture: add or remove layers
- Add L1 and/or L2 regularization
- Try different hyperparameters (such as the number of units per layer or learning rate) to find the optimized configuration
- Optionally, iterate on feature engineering: add new features, or remove features that doesn't seems informative