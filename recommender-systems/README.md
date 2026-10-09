# Recommender systems

Collaborative filtering, content-based and hybrid recommenders implemented from scratch, and a news recommender built on pre-trained text embeddings, all evaluated with offline protocols that mimic real use (temporal splits, ranking metrics).

| Notebook | Topics | Data |
|---|---|---|
| [01 · Collaborative filtering](01_collaborative_filtering.ipynb) | sparsity and the long tail, temporal split, bias baselines, user- and item-based kNN, matrix factorization with SGD, RMSE vs top-N metrics (precision, recall, NDCG, coverage, popularity) | MovieLens (100k ratings) |
| [02 · Content-based and hybrid](02_content_based_and_hybrid.ipynb) | cold start, TF-IDF item features and user profiles, data leakage from tags, per-user Naive Bayes, weighted and slot hybrids, limits of offline evaluation | MovieLens |
| [03 · News recommendation](03_news_recommendation.ipynb) | impression ranking, AUC/MRR/nDCG, popularity and category baselines, TF-IDF vs sentence-embedding user profiles, performance by history length | MIND small (Microsoft News) |

## Highlights

- **Better ratings, worse recommendations** (notebook 01): matrix factorization beats the baselines on RMSE but is the worst model at ranking, while item-based kNN gives the best balance of accuracy and personalization.
- **A leak that would have looked like progress** (notebook 02): user tags raise the AUC, but they are written after watching and leak future information.
- **Embeddings without training** (notebook 03): averaging pre-trained sentence embeddings of a user's clicked articles reaches AUC 0.64 on next-day news, half of which the model has never seen.

## Data

- **MovieLens** latest-small (F. Maxwell Harper and Joseph A. Konstan, 2015, *The MovieLens Datasets: History and Context*, ACM TiiS), included in `data/movielens/` under the GroupLens license (see its README).
- **MIND** (Wu et al., 2020, *MIND: A Large-scale Dataset for News Recommendation*, ACL) is distributed under the Microsoft Research License Terms and is not included: download *MINDsmall_train* and *MINDsmall_dev* from [msnews.github.io](https://msnews.github.io/) into `data/MIND/`.
