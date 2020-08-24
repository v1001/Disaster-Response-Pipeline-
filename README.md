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
