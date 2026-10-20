# 08 · Decision Trees and Random Forests

| Project | Notebook | Problem statement |
|---|---|---|
| Regression tree from scratch, then decision trees and random forests (grid search, feature importances) for sentiment classification of Amazon product reviews (bag-of-words) | [Decision_Trees_and_Random_Forests.ipynb](Decision_Trees_and_Random_Forests.ipynb) | [Problem_Statement.pdf](Problem_Statement.pdf) |

## Layout
| File | Role |
|---|---|
| `DecisionTreeRegressor.py` | `LeafNode` and `InternalDecisionNode` predict |
| `select_best_binary_split.py` | Best split search |
| `train_tree.py` | Recursive tree training |
| `tree_utils.py`, `pretty_print_sklearn_tree.py` | Helpers |
| `data_product_reviews/` | Train, validation and test splits plus vocabulary. `x_train.csv` is 94 MB, close to GitHub's 100 MB per-file limit |
