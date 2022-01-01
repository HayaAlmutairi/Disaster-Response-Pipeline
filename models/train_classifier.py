import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    load data from SQLite database.
    
    Arguments:
    database_filepath: Path to SQLite destination database (disaster_messages.db)
    
    Output:
    X: Features
    Y: Target
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("disaster_messages", con=engine)
    X = df['message']
    Y = df.iloc[:,4:]
    
    return X,Y


def tokenize(text):
    """
    Function to tokenize text  
    """
    
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]      
    return clean_tokens


def build_model():
    """
    Build the pipeline model
    Output:
    cv: Classifier
    """
    pipeline = Pipeline([
    ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
    ('tfidf_transformer', TfidfTransformer()),
    ('classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
    'classifier__estimator__n_estimators' : [50, 100]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test):
    """
    Evalueate the model performance. 
    Prints the classification report of each response variable.
    
    Arguments:
    model: ML Pipeline
    X_test: Test features
    Y_test: Test labels
    
    Output:
    Calling sklearn's classification_report on each column.
    """
    y_pred = model.predict(X_test)
    for idx, col in enumerate(Y_test):
        print(col, classification_report(Y_test.iloc[:,idx], y_pred[:,idx]))


def save_model(model, model_filepath):
    """
    Save the model in pickle file.
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()