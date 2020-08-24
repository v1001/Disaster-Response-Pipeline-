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
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_tweets', engine)
    X = df['message']
    y = df.drop(columns = ['id', 'message', 'original', 'genre'])
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text):
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
    for label in category_names:
        true = y_test[label].values
        pred = y_pred[label].values
        print('Classification result for category \"{}\":'.format(label))
        print(classification_report(true, pred))


def build_model():
    model_SVC = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize, ngram_range = (1,2), stop_words = 'english')),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LinearSVC(class_weight = 'balanced', random_state = 1)))
    ])
    return model_SVC

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = pd.DataFrame(model.predict(X_test), columns = category_names)
    show_results(Y_test, y_pred, category_names)


def save_model(model, model_filepath):
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
        model = build_model()
        
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