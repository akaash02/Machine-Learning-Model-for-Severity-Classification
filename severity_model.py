import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import os

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    data = pd.read_csv(file_path)
    print("Dataset Loaded. Initial rows:\n", data.head())
    return data

def clean_data(data):
    required_columns = ['annotation_severity', 'def_text', 'age', 'consensus_severity']
    for col in required_columns:
        if col not in data.columns:
            raise KeyError(f"The '{col}' column is missing from the dataset.")

    data = data.dropna(subset=['annotation_severity'])

    data['def_text'] = data['def_text'].fillna('')
    
    data['annotation_severity_encoded'] = data['annotation_severity'].astype('category').cat.codes
    data['consensus_severity_encoded'] = data['consensus_severity'].astype('category').cat.codes
    
    print("Data cleaned.")
    return data


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def preprocess_severity_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)

        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        words = word_tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(words)
    return ''

def preprocess_dataset(data):
    data['Processed Text'] = data['def_text'].apply(preprocess_text)
    data['Processed Annotation Severity'] = data['annotation_severity'].apply(preprocess_severity_text)
    data['Processed Consensus Severity'] = data['consensus_severity'].apply(preprocess_severity_text)
    
    print("Text preprocessing completed.")
    return data

def extract_features(data, max_features=10000):
    tfidf_def_text = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words='english')
    X_def_text = tfidf_def_text.fit_transform(data['Processed Text'])

    tfidf_annotation_severity = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words='english')
    X_annotation_severity = tfidf_annotation_severity.fit_transform(data['annotation_severity'])

    tfidf_consensus_severity = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words='english')
    X_consensus_severity = tfidf_consensus_severity.fit_transform(data['consensus_severity'])
    
    X_def_text_dense = X_def_text.toarray()
    X_annotation_severity_dense = X_annotation_severity.toarray()
    X_consensus_severity_dense = X_consensus_severity.toarray()

    X_combined = np.hstack((X_def_text_dense, X_annotation_severity_dense, X_consensus_severity_dense))
    
    y = data['annotation_severity_encoded']
    
    X_extra = data[['age', 'consensus_severity_encoded'] + [col for col in data.columns if col.startswith('VesselGroup_')]].values
    X_combined = np.hstack((X_combined, X_extra))
    
    print(f"TF-IDF + Extra Feature Shape: {X_combined.shape}")
    return X_combined, y, tfidf_def_text, tfidf_annotation_severity, tfidf_consensus_severity


# Balance data
def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("Data balanced using SMOTE.")
    return X_resampled, y_resampled

def select_features(X, y, k=2000):
    selector = SelectKBest(chi2, k=k)
    X_selected = selector.fit_transform(X, y)
    print(f"Reduced Feature Shape: {X_selected.shape}")
    return X_selected, selector

def preprocess_labels(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le

def build_nn_model(input_dim):
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_neural_network(X, y, batch_size=32, epochs=5):
    y, le = preprocess_labels(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = build_nn_model(X_train.shape[1])

    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
              validation_data=(X_val, y_val), verbose=1, callbacks=[checkpoint])

    best_model = tf.keras.models.load_model('best_model.h5')
    y_pred = best_model.predict(X_val)
    y_pred = np.argmax(y_pred, axis=1)

    print("Classification Report:\n", classification_report(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    return best_model


def main():
    file_path = 'data.csv'

    data = load_data(file_path)
    data = clean_data(data)
    data = preprocess_dataset(data)

    data = pd.get_dummies(data, columns=['VesselGroup'], drop_first=True)

    X, y, tfidf_def_text, tfidf_annotation_severity, tfidf_consensus_severity = extract_features(data)
    X, y = balance_data(X, y)
    X, selector = select_features(X, y)

    nn_model = train_neural_network(X, y)
    print("Model training complete.")

if __name__ == "__main__":
    main()
