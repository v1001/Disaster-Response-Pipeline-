import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_tweets', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    labeldf = pd.DataFrame(df.drop(columns = ['id', 'message', 'original', 'genre']).sum())
    labeldf.sort_values(by=[0], axis=0, inplace=True, ascending=False)
    
    labelhist = (df.drop(columns = ['id', 'message', 'original', 'genre']).sum(axis=1))
    hist = pd.cut(labelhist,[0,1,2,4,8,16,32]).value_counts()
    label_hist = hist.values.tolist()
    label_hist_names = ['0 - 1', '1 - 2', '2 - 4', '4 - 8', '8 - 16', '16 - 32']
    
    label_counts = [x[0] for x in labeldf.values.tolist()]
    label_names = labeldf.index.tolist()
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=label_names,
                    y=label_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Labels',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Label"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=label_hist_names,
                    y=label_hist
                )
            ],

            'layout': {
                'title': 'Number of Categories pro Message',
                'yaxis': {
                    'title': "Message coun"
                },
                'xaxis': {
                    'title': "Label Count"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

@app.route('/report')
def report():
    return render_template('classification_report.html')

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()