# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. If you are running the app localy go to http://0.0.0.0:3001/

### 1. Dataset
The dataset for this project comes from FigureEight and can be downloaded here: https://appen.com/datasets/combined-disaster-response-data/
The Overview states that "This dataset contains 30,000 messages drawn from events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, super-storm Sandy in the U.S.A. in 2012, and news articles spanning a large number of years and 100s of different disasters."

The dataset consists of twitter messages (text data fields for original language and english translation) and categories (36 binary fields with classification). There is also a "genre" field with multiple categories which is not used in this project.

### 2. ETL pipeline
ETL pipeline **extracts** messages and categories from csv files, **transforms** and cleans data and **loads** it to an SQLlite database. These are the cleaning steps:
- split the categories in separate fields and convert the values to numeric
- remove duplicate entries
- remove columns with no entries ("child alone")
- remove messages with "related" set to 2, which occur to have no other categories and are likely to contain errors, e.g. not translated text in the message field

### 3. ML pipeline
The goal of **machine learning pipeline** is to **transform** the input data, **train** the model and **predict** outputs of new samples. 

#### 3.1. Data trsansformation for natural language processing (NLP)
Natural language needs to be converted to a format, which machine learning algorithms can deal with. For this following transformation were applied to the text messages:
1. Only english letters were kept in the message. All other characters were replaced with spaces.
2. All words were converted to lower case.
3. The messages were tokenized - converted to lists of words.
4. *CountVectorizer* was used to create vocabulary over the text corpus and convert messages to vectors containing word counts.
5. *TF IDF* (term frequency inverse document frequency) transformer was used to convert word counts to TF-IDF.
6. As option word n-grams were used with *CountVectorizer* to account for word embeddings.
7. A custom transformer *NounsExtractor* was implemented with the goal to keep only nouns from the messages.

### 3.2. Training, cross-validation, hyperparameter optimization
To test different models and parameters a custom function *build_cv* was implemented. Following classifiers were examined in this project:
1. Multilayer perceptron
2. Random Forest
3. Multinomial Naive Bayes
4. Linear Support Vector Classifier
