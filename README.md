# Decision Trees Assignment Notebook

This notebook walks you through building and analysing decision trees for a heart disease classification problem.  
You’ll start by implementing core concepts like entropy and information gain, then use them to construct an ID3-style decision tree, extend it to numerical features, and finally explore overfitting, visualization, and random forests.

---

## ❤️ Dataset & Setup

**Goal:**  
Work with a real-world **heart disease** dataset and prepare it for training decision tree models.

### What Happens Here

- Load multiple heart disease files (`processed.va.data`, `processed.hungarian.data`, `processed.switzerland.data`, `processed.cleveland.data`) into a single `DataFrame`.
- Assign meaningful column names such as `age`, `sex`, `cp`, `trestbps`, `chol`, `thalach`, `ca`, `thal`, `num`, etc.
- Convert the multi-class target (`num`) into a **binary label** (e.g. “healthy” vs “heart disease”).
- Perform a **train/test split** (70/30) using `train_test_split`.
- Separate **categorical** and **numerical** features using a cardinality threshold (`cat_threshold`) and build:
  - `train_data_cat` / `test_data_cat` (categorical, integer-coded)
  - `train_data_num` / `test_data_num` (numerical, floats)

---

## 🧮 Assignment 1 — Entropy

**Goal:**  
Implement entropy as a measure of label impurity for binary classification.

### What to Implement

- `ratio(labels)`  
  - Compute the proportion \( p \) of `True` labels in a list/array of booleans.
- `entropy_sub(p)`  
  - Compute the contribution of a single probability \( p \) using base-2 logarithms.
  - Handle edge cases where \( p = 0 \) or \( p = 1 \) so you don’t take `log(0)`.
- `entropy(labels)`  
  - Use `ratio(labels)` and `entropy_sub` to compute the total entropy of the label set.

### What You’ll Check

- Use the provided `assert` statements to verify:
  - Entropy is **0** when all labels are the same.
  - Entropy is **1** when the labels are perfectly balanced (50/50).
- Generate lists of labels with different ratios of `True`/`False`, compute their entropy, and **plot entropy as a function of the positive-class ratio** to see the characteristic concave shape.

---

## 🧱 Assignment 2 — Splitting Data

**Goal:**  
Implement utilities to split a `DataFrame` by a categorical feature into subsets for each unique value.

### What to Implement

- `masks_for_split(dataframe, feature)`  
  - Return a dictionary mapping **feature value → boolean mask**, selecting rows where the column `feature` equals that value.
- `apply_split(dataframe, split_masks)`  
  - Given a `DataFrame` and the mask dictionary from `masks_for_split`, return a dictionary **feature value → subset DataFrame**.

### What You’ll Check

- Use the **PlayTennis** toy dataset (`tennis_data` and `tennis_labels`) to:
  - Inspect the masks for `Outlook`, `Temperature`, etc.
  - Apply the masks to split both data and labels.
- Confirm that each subset contains the correct rows (indices and values match expectations).

---

## 📉 Assignment 3 — Information Gain

**Goal:**  
Quantify how good a split is by measuring how much it reduces entropy.

### What to Implement

- `information_gain(split_labels)`  
  - Input: a dictionary mapping **feature value → label subset** (as produced by `apply_split` on the labels).
  - Compute:
    \[
    IG(S, A) = \phi(S) \;-\; \sum_{v \in Values(A)} \frac{|S_v|}{|S|} \phi(S_v)
    \]
    where:
    - \( S \) is the full label set for the node,
    - \( S_v \) are the label subsets for each split value,
    - \( \phi(\cdot) \) is the entropy.

### What You’ll Check

- Validate the implementation with the provided asserts, e.g.:
  - Information gain for splitting on `thal` and `ca` in the heart disease data.
- Conceptually: higher information gain → **better split** at that node.

---

## 🌳 Assignment 4 — ID3 Decision Tree (Categorical)

**Goal:**  
Build a full ID3-style decision tree that repeatedly splits on the feature with highest information gain.

### What to Implement (in `class DecisionTree`)

- **Initialization**
  - Store `data`, `labels`, and the minimum information-gain threshold `thres`.
  - Initialize fields:
    - `self.info_gain`
    - `self.split_feature`
    - `self.split_data`
    - `self.split_labels`
  - Call `self.determine_best_split()` to choose the best split at this node.
  - Mark `self.leaf` as `True` when `self.info_gain < self.thres`.
  - Store the **most common class** in `self.common_label` using `ratio(labels) >= 0.5`.

- `determine_best_split(self)`
  - Loop over all available features.
  - For each feature:
    - Create split masks → split data → split labels.
    - Compute information gain using `information_gain(...)`.
  - Choose the feature with **highest information gain**, and store:
    - `self.info_gain`
    - `self.split_feature`
    - `self.split_data`
    - `self.split_labels`

- `create_subtrees(self)`
  - If `self.leaf` is `False`, loop over all feature values in `self.split_data`:
    - Create subsets for data and labels.
    - Recursively construct `DecisionTree` subtrees.
    - Store each subtree in `self.branches[value]`.

- `classify(self, row)`
  - If the node is a leaf, return `self.common_label`.
  - Otherwise:
    - Look up the value of `self.split_feature` in `row`.
    - If that value is not in `self.branches`, fall back to `self.common_label`.
    - Else, forward the classification to the corresponding subtree’s `classify`.

- `validate(self, test_data, test_labels)`
  - Loop over all rows in `test_data`, classify each row, and compare to the true label.
  - Return the **classification accuracy**.

### What You’ll Check

- Build a tree on `train_data_cat` and evaluate:
  - Training accuracy on `(train_data_cat, train_labels)`.
  - Test accuracy on `(test_data_cat, test_labels)`.
- Inspect whether the tree **overfits** or generalizes well, and how `thres` affects it.

---

## 🔢 Assignment 5 — Numerical Decision Tree

**Goal:**  
Extend your decision-tree approach to handle **numerical features** (continuous values) in addition to categorical ones.

### What You’ll Do

- Use the earlier split of the dataset into:
  - `train_data_cat` (categorical features)
  - `train_data_num` (numerical features)
- Incorporate numerical splits by:
  - Evaluating candidate **thresholds** on numerical columns (e.g. midpoints between sorted unique values).
  - Treating each numerical split as a binary question:  
    “Is feature \( x_j \leq \tau \)?” vs “\( x_j > \tau \)”.
- Combine categorical and numerical features so that, at each node, the tree can consider **both types** of splits and choose the one with the highest information gain.

### What You’ll Check

- Train a numerical (or mixed-feature) decision tree on the heart disease data.
- Compare performance to the purely categorical tree:
  - Does including numerical splits improve accuracy?
  - How does tree depth change?

---

## 🛑 Assignment 6 — Preventing Overfitting

**Goal:**  
Explore regularization strategies for decision trees using scikit-learn and understand overfitting behavior.

### What You’ll Do

- Use `sklearn.tree.DecisionTreeClassifier` with the prepared features.
- Train a **fully grown** tree (no max depth) and evaluate:
  - Training accuracy.
  - Test accuracy.
- Then try limiting complexity with hyperparameters such as:
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`
- Compare how these settings affect:
  - Model complexity.
  - Train vs test accuracy (overfitting vs underfitting).

---

## 🌲 Assignment 7 — Plotting the Decision Tree

**Goal:**  
Visualize the structure of a trained decision tree.

### What You’ll Do

- Fit a `DecisionTreeClassifier` with a **limited depth** (small `max_depth`) and store it as `d_tree`.
- Use:
  - `from sklearn import tree`
  - `figure(figsize=(15, 10))`
  - `tree.plot_tree(...)`
- Configure:
  - `feature_names=list(df.columns.values)`
  - `class_names=['Healthy', 'Heart Disease']`
  - `impurity=False` to keep the tree readable.

### What You’ll Check

- Inspect the plot:
  - Which features appear near the root?
  - How do decision rules partition the data space?
- Adjust `max_depth` if the plot is too cluttered or too simple.

---

## 🌳 Assignment 8 — Random Forests

**Goal:**  
Reduce overfitting and improve robustness by building an ensemble of decision trees — a **Random Forest**.

### What to Implement

- `create_forest(n_trees)`
  - Return a list of **untrained** `DecisionTreeClassifier` instances (or your own tree class) that will form the forest.

- `train_forest(forest, data, labels, ratio=0.5)`
  - For each tree:
    - Draw a random **bootstrap sample** of the rows (e.g. using `DataFrame.sample`).
    - Optionally sample a random subset of features (“feature bagging”).
    - Train the tree on that sampled data/labels.

- `predict_forest(forest, data)`
  - For each sample:
    - Collect predictions from all trees in the forest.
    - Use **majority vote** over the binary predictions to get the final class.

### What You’ll Check

- Train a **Random Forest** with, for example:
  - `n_trees = 1000`
  - A subset of features per tree (e.g. 3 features).
- Compute train and test accuracy, and compare to a **single-tree** model:
  - Does the forest increase test accuracy?
  - Does it reduce variance and overfitting?

---

## ✅ Expected Learning Outcomes

By the end of this notebook, you should be able to:

- Explain and implement **entropy** and **information gain** for binary classification.
- Construct an **ID3 decision tree** from scratch for categorical features.
- Extend decision trees to handle **numerical** features via threshold-based splits.
- Diagnose and mitigate **overfitting** using depth limits and other regularization options.
- Visualize a decision tree using `sklearn.tree.plot_tree`.
- Implement and evaluate a simple **Random Forest** via bagging and majority voting.

---

## 🧷 Notes

- Make sure the following libraries are available:  
  `pandas`, `numpy`, `scikit-learn`, `matplotlib`.
- The notebook uses heart disease data from multiple sources; keep the `data/` folder next to the notebook.
- Cells marked with `# YOUR CODE HERE` are required for passing the automatic checks (`notebook_checker`).
- Rerun tests and assertions after each implementation step to catch bugs early.
