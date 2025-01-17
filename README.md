# Machine Learning Model for Severity Classification

This project focuses on building a machine learning model to classify the severity of a given annotation based on textual data. It leverages natural language processing (NLP) techniques, feature extraction, data balancing, and neural network training using Keras.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Extraction](#feature-extraction)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Running the Code](#running-the-code)

## Project Overview

The goal of this project is to build a classification model that predicts the severity of an annotation based on text data. The following steps are involved:

- Loading and cleaning the dataset.
- Preprocessing the text data.
- Feature extraction using TF-IDF (Term Frequency-Inverse Document Frequency).
- Balancing the data using SMOTE (Synthetic Minority Over-sampling Technique).
- Selecting the best features using SelectKBest.
- Building and training a neural network model.
- Evaluating the model using classification metrics such as the classification report and confusion matrix.

## Requirements

To run this project, you need the following Python libraries:

- `pandas` - For data manipulation.
- `numpy` - For numerical operations.
- `scikit-learn` - For machine learning algorithms and feature selection.
- `imblearn` - For handling imbalanced data using SMOTE.
- `nltk` - For text preprocessing.
- `tensorflow` - For building and training the neural network model.

Install the required libraries by running:

```bash
pip install pandas numpy scikit-learn imbalanced-learn nltk tensorflow
```

Additionally, the `nltk` package requires downloading some data files. These will be automatically downloaded when running the script.

## Data Preprocessing

1. **Loading the Dataset**:
   The dataset is loaded from a CSV file. The required columns are:
   - `annotation_severity` (target column)
   - `def_text` (text data for feature extraction)
   - `age` (numeric feature)
   - `consensus_severity` (numeric feature)

2. **Cleaning the Data**:
   - Missing `annotation_severity` values are dropped.
   - Missing text in `def_text` is filled with an empty string.
   - Categorical columns (`annotation_severity` and `consensus_severity`) are encoded into numeric values.

3. **Text Preprocessing**:
   The `def_text`, `annotation_severity`, and `consensus_severity` columns are preprocessed by:
   - Converting text to lowercase.
   - Removing non-alphabetic characters.
   - Tokenizing the text and removing stopwords.
   - Lemmatizing the words.

## Feature Extraction

Features are extracted from the text data using the **TF-IDF** method:
- `TfidfVectorizer` is applied to the `def_text`, `annotation_severity`, and `consensus_severity` columns to convert text into numeric features.
- The features are then combined with additional numerical features (`age`, `consensus_severity_encoded`) and one-hot encoded categorical variables (`VesselGroup`).

## Model Training

1. **Data Balancing**:
   Since the dataset may be imbalanced, the **SMOTE** technique is applied to generate synthetic samples for the minority class.

2. **Feature Selection**:
   **SelectKBest** is used to select the top `k` features based on the chi-squared test.

3. **Building the Neural Network**:
   A neural network model is built using Keras with the following layers:
   - Input layer with 512 units and ReLU activation.
   - Dropout layer with 30% dropout rate.
   - Hidden layer with 256 units and ReLU activation.
   - Output layer with 3 units and softmax activation for multi-class classification.

4. **Model Training**:
   The model is trained using the Adam optimizer and sparse categorical cross-entropy loss function. A **ModelCheckpoint** callback is used to save the best model based on validation accuracy.

## Evaluation

After training, the model is evaluated on the validation set using the following metrics:
- **Classification Report**: Provides precision, recall, and F1-score for each class.
- **Confusion Matrix**: Shows the true vs predicted class counts.

## Running the Code

1. **Prepare the Dataset**:
   Place your dataset (CSV file) in the same directory as the script, or update the `file_path` variable to point to your dataset's location.

2. **Run the Script**:
   Execute the script to train and evaluate the model.

```bash
python severity_classification.py
```

3. **Output**:
   The script will print the classification report and confusion matrix to the console and save the best model as `best_model.h5`.

---

## File Structure

```
severity_classification.py      # Main Python script
data.csv                       # Dataset (place your own CSV here)
best_model.h5                  # Saved model (generated after training)
README.md                      # This README file
```

---

## Notes

- Ensure that the dataset contains the required columns: `annotation_severity`, `def_text`, `age`, and `consensus_severity`.
- You may need to adjust the hyperparameters (e.g., batch size, epochs, k for feature selection) based on your specific dataset.
