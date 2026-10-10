# Movie Review Sentiment Classification with RNNs

> **School project.** Coursework for ST1504 Deep Learning (CA1, Part B), Diploma in Applied AI & Analytics, Singapore Polytechnic. Kept here as a record of my early deep learning work.

Classifies movie reviews as positive, neutral or negative using recurrent neural networks in Keras.

## What's in the notebook

- EDA: review length distributions, language detection, score distributions
- Cleaning: Malay reviews translated to English, rare-language rows dropped, labels checked against a polarity model and corrected
- Two-stage text augmentation (synonym replacement and contextual word embeddings) to balance the classes
- A grid search over RNN architectures (including LSTM, bidirectional LSTM and stacked GRU), optimisers, learning rates and dropout, ranked by macro F1 and accuracy

## Result

The best configurations reached roughly 90–92% test accuracy with macro F1 around 90%. The top run (RMSprop, learning rate 0.0005, dropout 0.5) scored 91.6% test accuracy, and its weights are the `.h5` file in this repo. SGD-based runs failed to converge for every architecture.

## Files

- `FINAL CA1 Part B.ipynb`: the full notebook with outputs
- `FINAL CA1 Part B.html`: the same notebook exported to HTML
- `M5_sparse_categorical_crossentropy_rmsprop_lr0.0005_do0.5.h5`: saved weights of the selected model

Stack: Python, TensorFlow/Keras, NLTK, nlpaug, Hugging Face Transformers, scikit-learn.
