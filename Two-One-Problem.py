"""
Two-One-Problem: Fix ONE from graph (imbalance)
===============================================

This lab demonstrates how to handle class imbalance in machine learning datasets
using RandomOverSampler from the imbalanced-learn library.

The code addresses the common problem where one class significantly outnumbers
another, leading to biased model performance.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings('ignore')

class TwoOneProblem:
    """
    Two-One-Problem: A comprehensive lab for demonstrating class imbalance handling techniques.
    Addresses the "Fix ONE from graph (imbalance)" challenge.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the lab with a random state for reproducibility.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_vec = None
        self.X_test_vec = None
        self.X_res = None
        self.y_res = None
        self.models = {}
        self.results = {}
        
    def create_imbalanced_dataset(self, n_samples=2000, n_features=20, 
                                 weights=[0.9, 0.1], flip_y=0.01):
        """
        Create an imbalanced synthetic dataset for demonstration.
        
        Args:
            n_samples (int): Total number of samples
            n_features (int): Number of features
            weights (list): Class distribution weights
            flip_y (float): Fraction of samples whose class is randomly flipped
            
        Returns:
            tuple: X, y features and labels
        """
        print("Creating imbalanced dataset...")
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            weights=weights,
            flip_y=flip_y,
            random_state=self.random_state
        )
        
        # Display class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")
        print(f"Imbalance ratio: {max(counts)/min(counts):.2f}:1")
        
        return X, y
    
    def create_text_dataset(self):
        """
        Create a sample text dataset for demonstration with TF-IDF vectorization.
        """
        print("Creating text dataset...")
        
        # Sample text data with imbalance
        positive_texts = [
            "I love this product, it's amazing!",
            "Excellent service and quality",
            "Best purchase I've made this year",
            "Outstanding experience overall",
            "Highly recommend to everyone",
            "Fantastic quality and value",
            "Perfect for my needs",
            "Exceeded my expectations completely"
        ] * 100  # 800 samples
        
        negative_texts = [
            "Terrible product, waste of money",
            "Poor quality and bad service",
            "Disappointed with this purchase",
            "Would not recommend to anyone",
            "Complete waste of time",
            "Worst experience ever",
            "Not worth the price",
            "Very unsatisfied customer"
        ] * 25  # 200 samples
        
        # Combine and create labels
        texts = positive_texts + negative_texts
        labels = [1] * len(positive_texts) + [0] * len(negative_texts)
        
        # Shuffle
        combined = list(zip(texts, labels))
        np.random.shuffle(combined)
        texts, labels = zip(*combined)
        
        print(f"Text dataset created with {len(texts)} samples")
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")
        
        return list(texts), list(labels)
    
    def split_data(self, X, y, test_size=0.2):
        """
        Split data into training and testing sets.
        
        Args:
            X: Features
            y: Labels
            test_size (float): Proportion of data for testing
        """
        print("Splitting data into train/test sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
        # Display class distribution in splits
        print("\nTraining set class distribution:")
        unique, counts = np.unique(self.y_train, return_counts=True)
        print(f"  {dict(zip(unique, counts))}")
        
        print("Test set class distribution:")
        unique, counts = np.unique(self.y_test, return_counts=True)
        print(f"  {dict(zip(unique, counts))}")
    
    def vectorize_text_data(self, max_features=1000):
        """
        Vectorize text data using TF-IDF.
        
        Args:
            max_features (int): Maximum number of features
        """
        if self.X_train is None or not isinstance(self.X_train[0], str):
            raise ValueError("Text data not loaded or not in correct format")
        
        print("Vectorizing text data with TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.X_train_vec = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vec = self.vectorizer.transform(self.X_test)
        
        print(f"Vectorized data shape - Train: {self.X_train_vec.shape}, Test: {self.X_test_vec.shape}")
    
    def apply_random_oversampling(self):
        """
        Apply RandomOverSampler to handle class imbalance.
        
        This is the core technique from the provided code snippet.
        """
        print("\n" + "="*50)
        print("APPLYING RANDOM OVERSAMPLING")
        print("="*50)
        
        # Determine which data to use (vectorized text or regular features)
        X_data = self.X_train_vec if hasattr(self, 'X_train_vec') and self.X_train_vec is not None else self.X_train
        
        print(f"Original training data shape: {X_data.shape}")
        print(f"Original class distribution:")
        unique, counts = np.unique(self.y_train, return_counts=True)
        print(f"  {dict(zip(unique, counts))}")
        
        # Apply RandomOverSampler (the core code from the snippet)
        ros = RandomOverSampler(random_state=self.random_state)
        self.X_res, self.y_res = ros.fit_resample(X_data, self.y_train)
        
        print(f"\nAfter oversampling:")
        print(f"Resampled data shape: {self.X_res.shape}")
        print(f"Resampled class distribution:")
        unique, counts = np.unique(self.y_res, return_counts=True)
        print(f"  {dict(zip(unique, counts))}")
        
        return self.X_res, self.y_res
    
    def train_models(self, use_resampled=True):
        """
        Train multiple models and compare performance.
        
        Args:
            use_resampled (bool): Whether to use resampled data for training
        """
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        # Determine training data
        if use_resampled and self.X_res is not None:
            X_train_data = self.X_res
            y_train_data = self.y_res
            data_type = "Resampled"
        else:
            X_train_data = self.X_train_vec if hasattr(self, 'X_train_vec') and self.X_train_vec is not None else self.X_train
            y_train_data = self.y_train
            data_type = "Original"
        
        print(f"Training with {data_type} data...")
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=self.random_state, n_estimators=100),
            'SVM': SVC(random_state=self.random_state, probability=True)
        }
        
        # Determine test data
        X_test_data = self.X_test_vec if hasattr(self, 'X_test_vec') and self.X_test_vec is not None else self.X_test
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_data, y_train_data)
            
            # Predictions
            y_pred = model.predict(X_test_data)
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, output_dict=True)
            
            # Store results
            self.models[f"{name}_{data_type}"] = model
            self.results[f"{name}_{data_type}"] = {
                'accuracy': accuracy,
                'precision_0': report['0']['precision'],
                'recall_0': report['0']['recall'],
                'f1_0': report['0']['f1-score'],
                'precision_1': report['1']['precision'],
                'recall_1': report['1']['recall'],
                'f1_1': report['1']['f1-score'],
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score (Class 1): {report['1']['f1-score']:.4f}")
    
    def compare_results(self):
        """
        Compare performance between original and resampled data.
        """
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        
        # Create comparison table
        comparison_data = []
        
        for key, metrics in self.results.items():
            comparison_data.append({
                'Model': key.replace('_Original', '').replace('_Resampled', ''),
                'Data_Type': 'Original' if '_Original' in key else 'Resampled',
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision_Class_1': f"{metrics['precision_1']:.4f}",
                'Recall_Class_1': f"{metrics['recall_1']:.4f}",
                'F1_Class_1': f"{metrics['f1_1']:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # Calculate improvements
        print("\n" + "-"*40)
        print("IMPROVEMENT ANALYSIS")
        print("-"*40)
        
        models = set([item['Model'] for item in comparison_data])
        
        for model in models:
            original = next((item for item in comparison_data 
                           if item['Model'] == model and item['Data_Type'] == 'Original'), None)
            resampled = next((item for item in comparison_data 
                            if item['Model'] == model and item['Data_Type'] == 'Resampled'), None)
            
            if original and resampled:
                acc_improvement = (float(resampled['Accuracy']) - float(original['Accuracy'])) * 100
                f1_improvement = (float(resampled['F1_Class_1']) - float(original['F1_Class_1'])) * 100
                
                print(f"\n{model}:")
                print(f"  Accuracy improvement: {acc_improvement:+.2f}%")
                print(f"  F1-Score (Class 1) improvement: {f1_improvement:+.2f}%")
        
        return df_comparison
    
    def visualize_results(self):
        """
        Create visualizations of the results.
        """
        print("\nGenerating visualizations...")
        
        # Prepare data for plotting
        plot_data = []
        for key, metrics in self.results.items():
            model_name = key.replace('_Original', '').replace('_Resampled', '')
            data_type = 'Original' if '_Original' in key else 'Resampled'
            plot_data.append({
                'Model': model_name,
                'Data_Type': data_type,
                'Accuracy': metrics['accuracy'],
                'F1_Class_1': metrics['f1_1']
            })
        
        df_plot = pd.DataFrame(plot_data)
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        sns.barplot(data=df_plot, x='Model', y='Accuracy', hue='Data_Type', ax=axes[0])
        axes[0].set_title('Model Accuracy Comparison')
        axes[0].set_ylim(0, 1)
        
        # F1-Score comparison
        sns.barplot(data=df_plot, x='Model', y='F1_Class_1', hue='Data_Type', ax=axes[1])
        axes[1].set_title('F1-Score (Class 1) Comparison')
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('C:\\Users\\Saboor\\CascadeProjects\\imbalance_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualization saved as 'imbalance_results.png'")
    
    def run_complete_lab(self, dataset_type='text'):
        """
        Run the complete lab pipeline.
        
        Args:
            dataset_type (str): 'text' or 'numeric' dataset type
        """
        print("="*60)
        print("CLASS IMBALANCE HANDLING LAB")
        print("="*60)
        
        if dataset_type == 'text':
            # Text dataset pipeline
            X, y = self.create_text_dataset()
            self.split_data(X, y)
            self.vectorize_text_data()
        else:
            # Numeric dataset pipeline
            X, y = self.create_imbalanced_dataset()
            self.split_data(X, y)
        
        # Train with original data
        self.train_models(use_resampled=False)
        
        # Apply oversampling and retrain
        self.apply_random_oversampling()
        self.train_models(use_resampled=True)
        
        # Compare results
        comparison = self.compare_results()
        
        # Visualize
        try:
            self.visualize_results()
        except Exception as e:
            print(f"Visualization failed: {e}")
        
        return comparison

def main():
    """
    Main function to run the Two-One-Problem lab demonstration.
    """
    print("Starting Two-One-Problem Lab...")
    
    # Initialize lab
    lab = TwoOneProblem(random_state=42)
    
    # Run with text dataset (demonstrates the provided code snippet)
    print("\n" + "="*60)
    print("DEMONSTRATION WITH TEXT DATASET")
    print("="*60)
    
    # This demonstrates the exact code snippet provided:
    # ros = RandomOverSampler(random_state=42)
    # X_res, y_res = ros.fit_resample(X_train_vec, y_train)
    
    results = lab.run_complete_lab(dataset_type='text')
    
    print("\n" + "="*60)
    print("TWO-ONE-PROBLEM LAB COMPLETED!")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. RandomOverSampler effectively balances class distribution")
    print("2. Balanced data typically improves minority class performance")
    print("3. The improvement varies by model type")
    print("4. F1-Score for minority class often shows significant gains")
    print("5. Overall accuracy may slightly decrease but model becomes more balanced")
    
    return lab, results

if __name__ == "__main__":
    lab, results = main()
