import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import prepare_data, clean_data, preprocess_features

# Create directories for storing results
def create_directories():
    """Create necessary directories for storing results."""
    directories = ['models', 'plots/confusion_matrix', 'plots/roc_curves']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def analyze_data(df: pd.DataFrame):
    """Analyze the dataset and print insights."""
    print("\nData Analysis:")
    print("-" * 50)
    
    # Analyze categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col} Distribution:")
        print(df[col].value_counts(normalize=True).round(3) * 100, "%")
    
    # Analyze numerical columns
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    print("\nNumerical Features Statistics:")
    print(df[numerical_cols].describe())
    
    # Print class distribution
    print("\nChurn Distribution:")
    print(df['Churn'].value_counts(normalize=True).round(3) * 100, "%")

def save_plot(plt, plot_type, model_name):
    """Save plot to appropriate directory."""
    plot_dir = f'plots/{plot_type}'
    filename = f'{model_name.lower().replace(" ", "_")}.png'
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()

def train_evaluate_model(model, name, X_train, X_test, y_train, y_test):
    """Train and evaluate a single model."""
    print(f"\n{name} Evaluation:")
    print("-" * 50)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"{name} accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False
    )
    plt.title(f"{name} Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    save_plot(plt, 'confusion_matrix', name)
    
    # Plot ROC curve if model supports predict_proba
    if hasattr(model, "predict_proba"):
        plt.figure(figsize=(8, 6))
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} ROC Curve')
        plt.legend()
        save_plot(plt, 'roc_curves', name)
    
    return model, accuracy

def main():
    # Create directories
    create_directories()
    
    # Load and clean data
    print("Loading and cleaning data...")
    df = pd.read_csv('Dataset/Telecom churn.csv')
    df_cleaned = clean_data(df)
    
    # Analyze data
    analyze_data(df_cleaned)
    
    # Prepare features and target
    print("\nPreparing features...")
    X, scaler = prepare_data(df_cleaned)
    y = (df_cleaned['Churn'] == 'Yes').astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize models
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=11),
        'SVM': SVC(random_state=1, probability=True),
        'Random Forest': RandomForestClassifier(
            n_estimators=500,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
            random_state=50,
            class_weight='balanced'
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced'
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    }
    
    # Dictionary to store results
    results = {}
    best_accuracy = 0
    best_model = None
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        trained_model, accuracy = train_evaluate_model(
            model, name, X_train, X_test, y_train, y_test
        )
        results[name] = accuracy
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = trained_model
    
    # Train Voting Classifier
    print("\nTraining Voting Classifier...")
    voting_clf = VotingClassifier(
        estimators=[
            ('gbc', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)),
            ('lr', LogisticRegression(max_iter=1000, class_weight='balanced')),
            ('rf', RandomForestClassifier(n_estimators=200, max_features='sqrt', class_weight='balanced'))
        ],
        voting='soft'
    )
    
    trained_model, accuracy = train_evaluate_model(
        voting_clf, "Voting Classifier", X_train, X_test, y_train, y_test
    )
    results["Voting Classifier"] = accuracy
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = trained_model
    
    # Print final results
    print("\nModel Accuracies:")
    print("-" * 50)
    for name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {accuracy:.4f}")
    
    # Save best model and scaler
    print(f"\nSaving best model: {max(results, key=results.get)}")
    joblib.dump(best_model, 'models/best_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    print("\nTraining completed!")
    print("Best model saved as 'models/best_model.joblib'")
    print("Scaler saved as 'models/scaler.joblib'")
    print("\nPlots have been saved in:")
    print("- plots/confusion_matrix/")
    print("- plots/roc_curves/")

if __name__ == "__main__":
    main()