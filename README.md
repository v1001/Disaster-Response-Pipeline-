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

#### 3.2. Training, cross-validation, hyperparameter optimization
First the data was split into **training** and **testing** data.
The data has multiple categories, hence a multi-output classifier was implemented.
Following base classifiers were examined:
1. Multilayer perceptron (MLP)
2. Random Forest (RF)
3. Multinomial Naive Bayes (NB)
4. Linear Support Vector Classifier (SVC)

#### 3.3. Comparing classifiers
The following table contains the ROC AUC score, training time and pickle size of the saved model.

Classifier| ROC AUC | size, MB | training time, s
------------ | ------------- | ------------- | -------------
Linear SVC | 0.703731 | 37.5 | 36.756851
MLPC | 0.659678 | 1590.0 | 1393.571046
RF | 0.580703 | 1020.0 | 532.916047
Multinomial NB | 0.647583 | 29.8 | 9.396868

#### 3.4. Cross-validation and grid search for MLPC
To test different models and parameters a custom function *build_cv* was implemented.

Cross-validation was performed for MLPC using grid search. Optimal parameters were applied to continue training. Training multi-output MLP classifier is essentially training one classifiers for each output. For this reason only one category "related" was used to perform grid search.

MLP classifier was first considered to be the most promising, because intuitively neural networks are considered to be the best for natural language processing. The first results were very disappointing: the classifier was overfitting, didn't do well with minority classes and took extremely long to train. Several steps were done to find optimal parameters:
* grid search for the best activation function and solver combination (relu, adam)
* grid search for the optimal initial learning rate and tolerance (0.001, 0.01)
* grid search for the optimal hidden layer structure (single layer, 55 neurons)
* several settings for *CountVectorizer* were tested (keeping stop words improved the results)
* configurations with and without TFIDF as well as different parameters were tested (not using TFIDF improved the results)

Early stopping was activated and validation fraction set to 10%. With this configuration the performance increased, but not for categories with strong imbalance.

#### 3.5. Further investigations
##### Neural networks for NLP
After studying literature about NLP it was obvious, that MLP is not the best neural network for the task. LSTM or RNN is preferred, because they naturally have the capacity to store information about previously processed information.
##### Variations in pipelines
It is worth to be noted, that not all ML pipelines used the same steps before classifiers were applied. For example only Linear SVC pipeline uses TF-IDF. This isn't however the reason for better performance. Other pipelines were validated with or without TF-IDF and had no increase or even a slight decrease in performance. Word n-grams of order 2 were used with Linear SVC. These also were tried with other classifiers, but they didn't improve their performance significantly. Removing stop words have shown almost no effect for all classifiers except MLP, which had slight performance increase when keeping the stop words.
##### Using only nouns
Count vectorizer produces a sparse matrix with over 25000 parameters. This dimension increases in factorial ration if n-grams are added for word embeddings. It might be useful to reduce the number of individual words for some cases. For the text corpus in this project the vocabulary size including 2-word n-grams was almost 200000.

A custom transformer *NounsExtractor* was implemented to extract nouns from the messages. The number of individual words in vocabulary was reduced to around 16000 and the vocabulary including 2-word n-grams was almost halved.

With this transformer the Linear SVC pipeline was validated. Here are the scores with and without *NounsExtractor*:
score | full text | only nouns
----------|----------|-----------
precision | 0.727760 | 0.686758
recall | 0.703731 | 0.675585
f1 | 0.707707 | 0.677456
roc_auc | 0.703731 | 0.675585

### 4. Closer look at the data, dealing with imbalance
The messages can be related to disaster or not:

![Histogram related messages distribution](/images/RelatedHistogram.PNG)

Related messages constitute around 76% of the cleaned data and are clearly the majority class.

Each related message can have multiple categories

![Number of message categories distribution](/images/MessageCategoriesHistogram.PNG)

Notice that this is a logarithmic plot on the y-axis. Most messages have less than 10 categories, but some can have up to 25.

Although the proportion of related messages is high and each related message has on average 4.17 categories some categories are drastically underrepresented in this dataset.

![Number of message categories distribution](/images/CategoriesMeanHistogram.PNG)

To deal with imbalance following measures were applied:
1. To validate the performance of ML pipeline following scoring metrics were applied:
    * precision (true positives / (true positives + false positives) )
    * recall (true positives  / (true positives + false negatives) )
    * f1 score (2 * (precision * recall) / (precision + recall) )
    * area under the reciever operating characteristic curve (ROC AUC) https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
2. To ensure that the influence of minority class is not shown in metrics macro averaging was applied to calculate composite score for every category as well as the overall score for a classifier. Macro averaging assigns equal weights to all classes so that if the score of a minority class has same impact as that of the majority class.
3. For MLP classifier upsampling of minority class was applied on training data.

### 5. Further thoughts on disaster response pipeline

A real disaster response pipeline has to deal with millions of messages. The fraction of related messages is much lower than in the presented dataset. For this scenario the following approach can be of advantage:
1. Create a classifier, which filters messages related to current disaster. This classifier must use additional data such as IP address or device location if possible. For training the NLP part of such a classifier a balanced dataset of related and unrelated messages is needed.
2. Create a classifier for coarse categories such as food, water, weather, earthquake etc.
3. Create a sub-categories classifier.
4. Extract proper nouns from messages to indicate names, addresses or landmarks.
