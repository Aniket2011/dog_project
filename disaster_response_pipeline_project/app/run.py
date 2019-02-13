import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Bar, Heatmap, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)
# function tokenizes the text 
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

#function to find the length of text
def compute_text_length(data):
    return np.array([len(text) for text in data]).reshape(-1, 1)

# load data from sqlite
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load the saved model classifier.pkl
model = joblib.load("../models/classifier.pkl")
df['text_length'] = compute_text_length(df['message'])

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']  # count of different genre
    genre_names = list(genre_counts.index)                 # names of gernes
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    category_map = df.iloc[:,4:].corr().values       # value of the category
    category_names = list(df.iloc[:,4:].columns)     # Category names

    # extract length of texts
    length_direct = df.loc[df.genre=='direct','text_length']    # The length for direct genre
    length_social = df.loc[df.genre=='social','text_length']    # length for social genre
    length_news = df.loc[df.genre=='news','text_length']        # Length for news genre
    # Create a json for plotly graph
    graphs = [
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
        },

        {
            'data': [
                Heatmap(
                    x=category_names,
                    y=category_names[::-1],
                    z=category_map
                )    
            ],

            'layout': {
                'title': 'Heatmap of Categories'
            }
        },

        {
            'data': [
                Histogram(
                    y=length_direct,
                    name='Direct',
                    opacity=0.5
                ),
                Histogram(
                    y=length_social,
                    name='Social',
                    opacity=0.5
                ),
                Histogram(
                    y=length_news,
                    name='News',
                    opacity=0.5
                )
            ],

            'layout': {
                'title': 'Distribution of Text Length',
                'yaxis':{
                    'title':'Count'
                },
                'xaxis': {
                    'title':'Text Length'
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    #with open('../data/data.txt', 'w') as outfile:  
    #  json.dump(graphJSON, outfile) 
    
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


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()