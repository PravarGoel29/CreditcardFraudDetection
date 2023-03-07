import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score


# Load the credit card fraud dataset
def load_data():
    data = pd.read_csv('creditcard.csv')
    return data


# Visualize the class distribution of the dataset
def plot_class_distribution(data):
    fig, ax = plt.subplots()
    ax = sns.countplot(data['Class'])
    ax.set_title('Class Distribution of Credit Card Fraud Dataset')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    plt.show()


# Visualize the distribution of features in the dataset
def plot_feature_distribution(data):
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
    ax[0].set_title('Distribution of "Amount" Feature')
    sns.histplot(data['Amount'], ax=ax[0])
    ax[1].set_title('Distribution of "Time" Feature')
    sns.histplot(data['Time'], ax=ax[1])
    plt.show()


# Visualize the correlation between features in the dataset
def plot_correlation(data):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('Correlation Heatmap of Credit Card Fraud Dataset')
    sns.heatmap(data.corr(), cmap='coolwarm_r', annot_kws={'size': 10}, ax=ax)
    plt.show()


# Preprocess the data and split it into training and testing sets
def preprocess_data(data):
    # Scale the "Amount" and "Time" features
    scaler = StandardScaler()
    data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data['Time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))

    # Split the data into features and labels
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Balance the dataset using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# Train a machine learning model on the training set
def train_model(X_train, y_train):
    # Train a logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Train a decision tree classifier
    # model = DecisionTreeClassifier(random_state=42)
    # model.fit(X_train, y_train)

    # Train a random forest classifier
    # model = RandomForestClassifier(random_state=42)
    # model.fit(X_train, y_train)

    return model


# Make predictions on the testing set using the trained model
def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred


# Visualize the performance of the model using a confusion matrix
def plot_confusion_matrix(y_test, y_pred):
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    plt.show()

# Visualize the performance of the model using a classification report
def plot_classification_report(y_test, y_pred):
    print(classification_report(y_test, y_pred))

# Visualize the performance of the model using an ROC curve
def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc))
    ax.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    plt.show()

# Run the entire pipeline
def run_pipeline():
    # Load the data
    data = load_data()

    # Visualize the class distribution of the dataset
    plot_class_distribution(data)

    # Visualize the distribution of features in the dataset
    plot_feature_distribution(data)

    # Visualize the correlation between features in the dataset
    plot_correlation(data)

    # Preprocess the data and split it into training and testing sets
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Train a machine learning model on the training set
    model = train_model(X_train, y_train)

    # Make predictions on the testing set using the trained model
    y_pred = predict(model, X_test)

    # Visualize the performance of the model using a confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    # Visualize the performance of the model using a classification report
    plot_classification_report(y_test, y_pred)

    # Visualize the performance of the model using an ROC curve
    y_pred_proba = model.predict_proba(X_test)
    plot_roc_curve(y_test, y_pred_proba)

if __name__ == '__main__':
    run_pipeline()

