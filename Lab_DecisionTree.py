import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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


def load_wine_quality_data():
    """Load the Wine Quality dataset and preprocess"""
    wine_quality = fetch_ucirepo(id=186) 
    
    # Get features and targets
    feature_data = wine_quality.data.features
    label_data = wine_quality.data.targets 
    
    # Print dataset information for debugging
    print("\nDataset Metadata:")
    print(wine_quality.metadata)
    print("\nVariables Information:")
    print(wine_quality.variables)
    

    print("\nFeature columns:")
    print(feature_data.columns.tolist())
    print("\nTarget columns:")
    print(label_data.columns.tolist())
    
    
    label_data = label_data.iloc[:, 0]
    label_data = label_data.apply(lambda x: (
        0 if x <= 4 else
        1 if x <= 6 else
        2
    ))
    
    
    label_data.name = 'target'
    
    # Print the grouped target values for verification
    print("\nGrouped Target values:")
    print(label_data.value_counts())
    
    return feature_data, label_data


def load_car_quality_data():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    """Load the Car Quality dataset from UCI repository"""
    # Fetch dataset
    car_quality = fetch_ucirepo(id=19)  # Car Evaluation dataset ID on UCI repository

    # Convert to DataFrame (if not already)
    feature_data = car_quality.data.features
    label_data = car_quality.data.targets.copy()  # Make a copy to avoid SettingWithCopyWarning

    # Print dataset information for debugging
    print("\nDataset Metadata:")
    print(car_quality.metadata)  # Correct variable name
    print("\nVariables Information:")
    print(car_quality.variables)

    # Print the actual column names
    print("\nFeature columns:")
    print(feature_data.columns.tolist())
    print("\nTarget columns:")
    print(label_data.columns.tolist())

    # Convert target to numeric if needed
    print("\nTarget values:")
    print(label_data.iloc[:, 0].value_counts())  # Show the distribution of target values

    # Convert label_data to numeric using map
    label_data = label_data.iloc[:, 0].map({
        'unacc': 0,
        'acc': 1,
        'good': 2,
        'vgood': 3
    })
    label_data.name = 'target'

    # Print the converted target values for verification
    print("\nConverted Target values:")
    print(label_data.value_counts())

    # Encode categorical features to numeric values
    feature_data = encode_features(feature_data)

    return feature_data, label_data

from sklearn.preprocessing import LabelEncoder

def encode_features(feature_data):
    """Encode categorical features to numeric values"""
    le = LabelEncoder()
    for column in feature_data.columns:
        if feature_data[column].dtype == 'object':
            feature_data[column] = le.fit_transform(feature_data[column])
    return feature_data


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

def save_distribution_plots(original_labels, splits, dataset_name):
    """Save class distribution plots to files"""
    # Plot original distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x=original_labels)
    plt.title(f'Original Dataset Class Distribution ({dataset_name})')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.savefig(f'original_distribution_{dataset_name}.png')
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
        plt.title(f'Class Distribution for {split_name} Split ({dataset_name})')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.savefig(f'distribution_{split_name}_{dataset_name}.png')
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
    # Debug thông tin dữ liệu đầu vào
    print(f"Feature data shape: {X_train.shape}")
    print(f"Label data shape: {y_train.shape}")
    print(f"First few rows of X_train:\n{X_train.head() if isinstance(X_train, pd.DataFrame) else X_train[:5]}")
    print(f"First few rows of y_train:\n{y_train.head() if isinstance(y_train, pd.Series) else y_train[:5]}")

    # Đảm bảo dữ liệu không có NaN và định dạng đúng
    X_train = X_train.fillna(0) if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.dropna() if isinstance(y_train, pd.Series) else y_train
    X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.to_numpy().ravel() if isinstance(y_train, pd.Series) else y_train

    # Khởi tạo classifier
    clf = DecisionTreeClassifier(
        criterion='entropy',  # Use information gain
        random_state=42
    )

    # Fit model
    clf.fit(X_train, y_train)
    return clf

def visualize_tree(clf, feature_names, split_name, dataset_name):
    """Create and save decision tree visualization"""
    # Ensure class names are strings
    class_names = [str(cls) for cls in clf.classes_]

    dot_data = export_graphviz(
        clf,
        feature_names=feature_names,  
        class_names=class_names,  # Pass string class names here
        filled=True,
        rounded=True,
        special_characters=True,
        proportion=True
    )
    
    # Create and save graph
    graph = graphviz.Source(dot_data)
    graph.render(f'decision_tree_{split_name}_{dataset_name}', format='png', cleanup=True)
    
    return graph

def build_and_visualize_trees(splits, feature_names, dataset_name):
    """Build and visualize decision trees for all splits"""
    for split_name, split_data in splits.items():
        print(f"\nBuilding decision tree for {split_name} split ({dataset_name}):")
        
        # Build tree
        clf = build_decision_tree(
            split_data['feature_train'], 
            split_data['label_train']
        )
        
        # Visualize tree
        visualize_tree(clf, feature_names, split_name, dataset_name)
        
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


def evaluate_classifier(clf, X_test, y_test,height = None):
    """Evaluate the classifier and print the classification report and confusion matrix"""
    y_pred = clf.predict(X_test)
    print("\nEvaluation Results:",height, "Height")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return accuracy_score(y_test, y_pred)

def analyze_tree_depth(splits, feature_names, dataset_name):
    """Analyze the effect of tree depth on classification accuracy"""
    depths = [None, 2, 3, 4, 5, 6, 7]
    accuracies = []
    depth_labels = ['Unlimited' if depth is None else depth for depth in depths]

    for depth in depths:
        if depth is None:
            clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
        else:
            clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)
        clf.fit(splits['80_20']['feature_train'], splits['80_20']['label_train'])
        accuracy = evaluate_classifier(clf, splits['80_20']['feature_test'], splits['80_20']['label_test'],depth)
        accuracies.append(accuracy)

        try:
            dot_data = export_graphviz(
                clf,
                feature_names=feature_names,
                class_names=[str(cls) for cls in clf.classes_],
                filled=True,
                rounded=True,
                special_characters=True
            )
            graph = graphviz.Source(dot_data)
            graph.render(f"decision_tree_depth_{depth}_{dataset_name}", format='png', cleanup=True)
        except Exception as e:
            print(f"Visualization failed for depth {depth}: {e}")

    # Plot accuracy vs. tree depth
    plt.figure()
    plt.plot(depth_labels, accuracies, marker='o')
    plt.xlabel('Tree Depth')
    plt.ylabel('Accuracy')
    plt.title(f'Effect of Tree Depth on Accuracy ({dataset_name})')
    plt.grid(True)
    plt.savefig(f'accuracy_vs_depth_{dataset_name}.png')
    plt.show()
def main():
    datasets = {
        #'breast_cancer': load_breast_cancer_data,
        'wine_quality': load_wine_quality_data,
        #'car Quality': load_car_quality_data,
        # Add additional datasets here
    }

    for dataset_name, load_data_func in datasets.items():
        print(f"\nLoading {dataset_name} dataset from UCI repository...")
        features, labels = load_data_func()
        
        # Create splits
        print(f"\nCreating train/test splits for {dataset_name}...")
        splits = create_data_splits(features, labels)
        
        # Save distribution plots
        print(f"\nSaving class distribution plots for {dataset_name}...")
        save_distribution_plots(labels, splits, dataset_name)
        
        # Print split information
        print(f"\nSplit Information for {dataset_name}:")
        print_split_info(splits)
        
        # Build and visualize decision trees
        print(f"\nBuilding and visualizing decision trees for {dataset_name}...")
        build_and_visualize_trees(splits, features.columns, dataset_name)  # Fixed call
        
        # Evaluate decision tree classifiers
        print(f"\nEvaluating decision tree classifiers for {dataset_name}...")
        clf = build_decision_tree(splits['80_20']['feature_train'], splits['80_20']['label_train'])
        evaluate_classifier(clf, splits['80_20']['feature_test'], splits['80_20']['label_test'])
        
        # Analyze tree depth and accuracy
        print(f"\nAnalyzing tree depth and accuracy for {dataset_name}...")
        analyze_tree_depth(splits, features.columns, dataset_name)



if __name__ == "__main__":
    main()