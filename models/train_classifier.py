import sys
# import libraries
import re
import pickle
import numpy as np
import pandas as pd

# import nltk
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# import classfiers
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# import feature extraction, pipeline and model selection tools
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report, roc_auc_score

from sqlalchemy import create_engine

def load_data(database_filepath):
    """Saves DataFrame to a SQL database

    Keyword arguments:
    database_filpath -- file path to the file with database
    
    Return:
    X -- messages from database as pandas DataFrame
    y -- categories from database as pandas DataFrame
    category_names -- python list
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_tweets', engine)
    X = df['message']
    y = df.drop(columns = ['id', 'message', 'original', 'genre'])
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text):
    """transforms string into list with tokens (words)

    Keyword arguments:
    text -- string with text
    
    Return:
    tokens -- python list with tokens
    """
    # remove non-letter characters
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    # tokenize
    tokens = word_tokenize(text)
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
 
    return clean_tokens


def show_results(y_test, y_pred, category_names):
    """display validation results (scores)

    Keyword arguments:
    y_test -- true test labels as pandas DataFrame
    y_pred -- predicted test labels as pandas DataFrame
    category_names -- names of categories as python list
    
    Return:
    None

    """
    for label in category_names:
        true = y_test[label].values
        pred = y_pred[label].values
        print('Classification result for category \"{}\":'.format(label))
        print(classification_report(true, pred))

        
def build_cv(pipeline, parameters):
    """build grid search

    Keyword arguments:
    pipeline -- ML pipeline
    parameters -- python dictionary with parameters
    
    Return:
    GridSearchCV object
    """
    # create grid search object
    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 5, n_jobs = -1, scoring = 'f1_weighted', refit = True,
                     return_train_score = False)

    return cv


def build_model(X_train, Y_train):
    """build machine learning pipeline

    Keyword arguments:

    Return:
    Pipeline

    """
    model_SVC = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize, ngram_range = (1,2), stop_words = 'english')),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LinearSVC(class_weight = 'balanced', random_state = 1)))
    ])
    
    parameters = {
        'vect__ngram_range': [(1,1), (1,2)],
        'tfidf__use_idf': [True, False],
        'clf__estimator__fit_intercept': [True, False]
    }
    
    cv = build_cv(model_SVC, parameters)
    cv.fit(X_train, Y_train)
    
    return cv.best_estimator_

def evaluate_model(model, X_test, Y_test, category_names):
    """predict results on test data and display validation (scores)

    Keyword arguments:
    model -- pandas Pipeline
    y_test -- true test labels as pandas DataFrame
    X_test -- test data
    category_names -- names of categories as python list
    
    Return:
    None

    """
    y_pred = pd.DataFrame(model.predict(X_test), columns = category_names)
    show_results(Y_test, y_pred, category_names)


def save_model(model, model_filepath):
    """saves model to specified path as pickle

    Keyword arguments:
    model_filepath -- file path to model file
    
    Return:
    None

    """
    f = open(model_filepath,'wb')
    pickle.dump(model, f)
    f.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train, Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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