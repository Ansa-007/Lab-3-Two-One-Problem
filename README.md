# Two-One-Problem: Fix ONE from graph (imbalance)

A comprehensive Python laboratory demonstrating how to handle class imbalance in machine learning datasets using **RandomOverSampler** from the imbalanced-learn library. This lab addresses the "Fix ONE from graph (imbalance)" challenge.

## 🎯 Learning Objectives

- Understand the problem of class imbalance in machine learning
- Learn how to apply RandomOverSampler to balance datasets
- Compare model performance before and after oversampling
- Analyze the impact on different evaluation metrics
- Visualize performance improvements

## 📋 Prerequisites

### Required Libraries

Install the following packages before running the lab:

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn
```

### Library Versions (Recommended)

- `numpy >= 1.21.0`
- `pandas >= 1.3.0`
- `scikit-learn >= 1.0.0`
- `imbalanced-learn >= 0.8.0`
- `matplotlib >= 3.4.0`
- `seaborn >= 0.11.0`

## 🚀 Quick Start

### Basic Usage

```python
from Two-One-Problem import TwoOneProblem

# Initialize the lab
lab = TwoOneProblem(random_state=42)

# Run complete pipeline with text data
results = lab.run_complete_lab(dataset_type='text')

# Or run with numeric data
results = lab.run_complete_lab(dataset_type='numeric')
```

### 😀 Core Code Snippet (The Focus of This Lab)

```python
from imblearn.over_sampling import RandomOverSampler

# Initialize the oversampler
ros = RandomOverSampler(random_state=42)

# Apply to your training data (after vectorization for text data)
X_res, y_res = ros.fit_resample(X_train_vec, y_train)

# Retrain your model with the balanced data
model.fit(X_res, y_res)  # Expected: +5-10% performance boost
```

## 📊 Lab Features

### 1. **Dataset Generation**
- **Text Dataset**: Simulates real-world text classification with 90:10 imbalance
- **Numeric Dataset**: Synthetic classification data with configurable imbalance ratios

### 2. **Preprocessing Pipeline**
- Text vectorization using TF-IDF
- Train/test splitting with stratification
- Feature scaling for numeric data

### 3. **Oversampling with RandomOverSampler**
- Balances class distribution by duplicating minority class samples
- Maintains original data distribution
- Configurable random state for reproducibility

### 4. **Model Training & Evaluation**
- Multiple algorithms: Logistic Regression, Random Forest, SVM
- Comprehensive metrics: Accuracy, Precision, Recall, F1-Score
- Before/after comparison analysis

### 5. **Visualization**
- Performance comparison charts
- Confusion matrix visualizations
- Improvement analysis graphs

## 🔧 Detailed Usage

### Step-by-Step Pipeline

```python
# 1. Initialize lab
lab = TwoOneProblem(random_state=42)

# 2. Create or load data
X, y = lab.create_text_dataset()  # or create_imbalanced_dataset()

# 3. Split data
lab.split_data(X, y, test_size=0.2)

# 4. Vectorize (for text data)
lab.vectorize_text_data(max_features=1000)

# 5. Train models on original data
lab.train_models(use_resampled=False)

# 6. Apply oversampling
X_res, y_res = lab.apply_random_oversampling()

# 7. Train models on resampled data
lab.train_models(use_resampled=True)

# 8. Compare results
comparison = lab.compare_results()

# 9. Visualize results
lab.visualize_results()
```

### ⚙️ Custom Configuration

```python
# Create custom imbalanced dataset
X, y = lab.create_imbalanced_dataset(
    n_samples=5000,
    n_features=30,
    weights=[0.95, 0.05],  # 95:5 imbalance
    flip_y=0.02
)

# Custom text vectorization
lab.vectorize_text_data(
    max_features=2000,
    ngram_range=(1, 3)
)
```

## 📈 Expected Results

### Performance Improvements

When using RandomOverSampler, you can typically expect:

- **+5-10% boost** in minority class F1-Score
- **Improved recall** for the minority class
- **More balanced** precision/recall trade-off
- **Slight decrease** in overall accuracy (but better balance)

### Sample Output

```
PERFORMANCE COMPARISON
============================================================
Model              Data_Type  Accuracy  Precision_Class_1  Recall_Class_1  F1_Class_1
Logistic Regression Original   0.9250           0.8000          0.6000      0.6857
Logistic Regression Resampled  0.9125           0.7273          0.8000      0.7619
Random Forest      Original   0.9375           0.8571          0.6000      0.7059
Random Forest      Resampled  0.9250           0.7895          0.7500      0.7692
SVM                Original   0.9125           0.7273          0.8000      0.7619
SVM                Resampled  0.9250           0.8000          0.8000      0.8000

IMPROVEMENT ANALYSIS
----------------------------------------
Logistic Regression:
  Accuracy improvement: -1.25%
  F1-Score (Class 1) improvement: +7.62%

Random Forest:
  Accuracy improvement: -1.25%
  F1-Score (Class 1) improvement: +6.33%

SVM:
  Accuracy improvement: +1.25%
  F1-Score (Class 1) improvement: +3.81%
```

## 🎨 Visualizations

The lab generates comprehensive visualizations:

1. **Accuracy Comparison Bar Chart**: Shows performance before/after oversampling
2. **F1-Score Comparison**: Focuses on minority class performance
3. **Confusion Matrices**: Visual representation of classification results

All visualizations are saved as `imbalance_results.png` in the project directory.

## 🔍 Understanding the Code

### Key Components

1. **RandomOverSampler**: The core technique from your code snippet
   ```python
   ros = RandomOverSampler(random_state=42)
   X_res, y_res = ros.fit_resample(X_train_vec, y_train)
   ```

2. **Class Imbalance Detection**: Automatic analysis of class distribution
3. **Performance Metrics**: Comprehensive evaluation focusing on minority class
4. **Model Comparison**: Multiple algorithms to show consistent improvements

### Why RandomOverSampler?

- **Simple to implement**: Just 2 lines of code
- **Effective**: Consistently improves minority class performance
- **No information loss**: Unlike undersampling techniques
- **Model-agnostic**: Works with any classification algorithm

## 🛠️ Troubleshooting

### Common Issues

1. **Memory Error with Large Datasets**
   ```python
   # Reduce dataset size or max_features
   X, y = lab.create_imbalanced_dataset(n_samples=1000)  # Instead of 5000
   lab.vectorize_text_data(max_features=500)  # Instead of 1000
   ```

2. **Installation Issues**
   ```bash
   # Install imbalanced-learn separately
   pip install imbalanced-learn
   
   # For conda users
   conda install -c conda-forge imbalanced-learn
   ```

3. **Visualization Not Displaying**
   ```python
   # Ensure matplotlib backend is set
   import matplotlib
   matplotlib.use('Agg')  # For headless environments
   ```

### Performance Tips

1. **For Large Text Datasets**:
   ```python
   # Use limited features and n-grams
   lab.vectorize_text_data(max_features=500, ngram_range=(1, 1))
   ```

2. **For Faster Training**:
   ```python
   # Use simpler models or reduce estimators
   from sklearn.linear_model import LogisticRegression
   # Or reduce n_estimators in RandomForest
   ```

## 📚 Advanced Topics

### Alternative Oversampling Techniques

```python
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

# SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# ADASYN (Adaptive Synthetic Sampling)
adasyn = ADASYN(random_state=42)
X_res, y_res = adasyn.fit_resample(X_train, y_train)
```

### Combining Techniques

```python
from imblearn.combine import SMOTETomek, SMOTEENN

# SMOTE + Tomek links
smote_tomek = SMOTETomek(random_state=42)
X_res, y_res = smote_tomek.fit_resample(X_train, y_train)
```

## 🤝 Contributing

To extend this lab:

1. Add new oversampling techniques
2. Include more datasets
3. Add more visualization options
4. Implement cross-validation
5. Add hyperparameter tuning

## 📄 License

This educational lab is provided for learning purposes. Feel free to use and modify for your projects.

## 📞 Support

For questions or issues:

1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure you're using compatible library versions
4. Test with smaller datasets first

---

## 🎓 Happy Learning! 

Remember: The key insight is that **balanced data leads to more equitable model performance**, especially for minority classes that are often the most important in real-world applications.

---

## 👩‍🎓 Author

Generated by **Khansa Younas** for educational purposes to handle class imbalance in machine learning datasets using **RandomOverSampler** from the imbalanced-learn library.



