import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns

def load_breast_cancer_data():
    """Load the breast cancer dataset from UCI repository"""
    # Fetch dataset
    breast_cancer = fetch_ucirepo(id=17)
    
    # Get features and targets
    feature_data = breast_cancer.data.features
    label_data = breast_cancer.data.targets
    # Print dataset information for debugging
    print("\nDataset Metadata:")
    print(breast_cancer.metadata)
    print("\nVariables Information:")
    print(breast_cancer.variables)
    
    # Print the actual column names
    print("\nFeature columns:")
    print(feature_data.columns.tolist())
    print("\nTarget columns:")
    print(label_data.columns.tolist())
    
    # Convert target to numeric if needed
    # Assuming target is already in numeric format, but let's verify
    print("\nTarget values:")
    print(label_data.iloc[:, 0].value_counts())
    
    # Convert to series with name 'target'
    label_data = label_data.iloc[:, 0]
    label_data.name = 'target'
    
    return feature_data, label_data

def create_data_splits(features, labels):
    """Create train/test splits with different proportions"""
    test_sizes = [0.6, 0.4, 0.2, 0.1]
    all_splits = {}
    
    for test_size in test_sizes:
        train_size = 1 - test_size
        feature_train, feature_test, label_train, label_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=42,
            shuffle=True,
            stratify=labels
        )
        
        split_name = f"{int(train_size*100)}_{int(test_size*100)}"
        all_splits[split_name] = {
            'feature_train': feature_train,
            'feature_test': feature_test,
            'label_train': label_train,
            'label_test': label_test
        }
    
    return all_splits

def save_distribution_plots(original_labels, splits):
    """Save class distribution plots to files"""
    # Plot original distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x=original_labels)
    plt.title('Original Dataset Class Distribution')
    plt.xlabel('Class (0: Malignant, 1: Benign)')
    plt.ylabel('Count')
    plt.savefig('original_distribution.png')
    plt.close()
    
    # Plot distributions for each split
    for split_name, split_data in splits.items():
        plt.figure(figsize=(10, 6))
        plot_data = pd.DataFrame({
            'Class': pd.concat([split_data['label_train'], split_data['label_test']]),
            'Set': ['Train']*len(split_data['label_train']) + 
                  ['Test']*len(split_data['label_test'])
        })
        sns.countplot(data=plot_data, x='Class', hue='Set')
        plt.title(f'Class Distribution for {split_name} Split')
        plt.xlabel('Class (0: Malignant, 1: Benign)')

        plt.savefig(f'distribution_{split_name}.png')
        plt.close()

def print_split_info(splits):
    """Print information about each split"""
    for split_name, split_data in splits.items():
        print(f"\nSplit {split_name} (Train/Test):")
        print(f"Training set:")
        print(f"- Features shape: {split_data['feature_train'].shape}")
        print(f"- Number of features: {split_data['feature_train'].shape[1]}")
        print("\nClass distribution:")
        print("Train set:")
        print(split_data['label_train'].value_counts(normalize=True))
        print("\nTest set:")
        print(split_data['label_test'].value_counts(normalize=True))

def build_decision_tree(X_train, y_train):
    """Build decision tree classifier using information gain"""
    clf = DecisionTreeClassifier(
        criterion='entropy',  # Use information gain
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf

def visualize_tree(clf, feature_names, split_name):
    """Create and save decision tree visualization"""
    dot_data = export_graphviz(
        clf,
        feature_names=feature_names,  
        class_names=['malignant', 'benign'],
        filled=True,
        rounded=True,
        special_characters=True,
        proportion=True
    )
    
    # Create and save graph
    graph = graphviz.Source(dot_data)
    graph.render(f'decision_tree_{split_name}', format='png', cleanup=True)
    
    return graph

def build_and_visualize_trees(splits, feature_names):
    """Build and visualize decision trees for all splits"""
    for split_name, split_data in splits.items():
        print(f"\nBuilding decision tree for {split_name} split:")
        
        # Build tree
        clf = build_decision_tree(
            split_data['feature_train'], 
            split_data['label_train']
        )
        
        # Visualize tree
        visualize_tree(clf, feature_names, split_name)
        
        # Print tree information
        print(f"Number of nodes: {clf.tree_.node_count}")
        print(f"Tree depth: {clf.get_depth()}")
        
        # Print feature importance
        importances = pd.Series(
            clf.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False)
        
        print("\nTop 5 most important features:")
        print(importances.head())

def main():
    # Load data
    print("Loading breast cancer dataset from UCI repository...")
    features, labels = load_breast_cancer_data()
    
    # Create splits
    print("\nCreating train/test splits...")
    splits = create_data_splits(features, labels)
    
    # Save distribution plots
    print("\nSaving class distribution plots...")
    save_distribution_plots(labels, splits)
    
    # Print split information
    print("\nSplit Information:")
    print_split_info(splits)
    
    # Build and visualize decision trees (Phần mới thêm cho 2.2)
    print("\nBuilding and visualizing decision trees...")
    build_and_visualize_trees(splits, features.columns)

if __name__ == "__main__":
    main()