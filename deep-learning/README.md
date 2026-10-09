# Deep learning

Neural networks in PyTorch, from the training loop to Transformers. Every model is trained from scratch on a CPU, evaluated on held-out data and compared with sensible baselines.

| Notebook | Topics | Data |
|---|---|---|
| [01 · Gradient descent](01_gradient_descent.ipynb) | autograd, learning rate and stability (η < 2/λ_max), ill-conditioning, feature standardization, comparison with least squares | quadratic function, polynomial regression |
| [02 · Feed-forward networks](02_feedforward_networks.ipynb) | decision boundaries, depth and width, overfitting, validation monitoring, dropout, hyperparameter search | Gaussian classes, two moons, MNIST |
| [03 · Convolutional networks](03_convolutional_networks.ipynb) | convolutions vs fully connected layers, batch normalization, data augmentation, filters and feature maps, confusion analysis | CIFAR-10 |
| [04 · LSTM language model](04_lstm_language_model.ipynb) | tokenization and Zipf's law, perplexity vs n-gram baselines, decoding strategies (greedy, temperature, top-k), word embeddings | 35,000 US political news headlines |
| [05 · Transformer for the TSP](05_transformer_tsp.ipynb) | pointer attention for permutation outputs, set encoders, teacher forcing, sampling, combining learning with 2-opt local search | 52,000 random 20-city instances with optimal tours |

## Highlights

- **The learning rate has a sharp limit you can compute** (notebook 01): 0.03 converges, 0.033 diverges, exactly as 2/λ_max predicts.
- **Fewer parameters, much better accuracy** (notebook 03): a CNN with batch normalization and augmentation reaches 86% on CIFAR-10 with 6× fewer parameters than a fully connected network that stops at 56%.
- **Half the perplexity of a bigram model** (notebook 04), with an honest look at what greedy and sampled generations look like.
- **A Transformer that learns to route** (notebook 05): trained only on examples of optimal tours, it is 1.2% from the optimum with greedy decoding and finds the optimal tour on 93% of unseen instances when sampling.

## Running the notebooks

MNIST and CIFAR-10 are downloaded automatically on first run (CIFAR-10 from the Hugging Face hub, which requires the `datasets` package). The headlines and the TSP instances are included in `data/`. All notebooks run on a CPU; the longest (03, 04, 05) take between 20 and 45 minutes each on a 16-core machine.

The headlines are a subset of the *News Category Dataset* (R. Misra, 2022), released under CC BY 4.0.
