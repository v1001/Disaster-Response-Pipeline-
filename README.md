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
The dataset used for this project can be downloaded here: https://appen.com/datasets/combined-disaster-response-data/

The Overview states:
"This dataset contains 30,000 messages drawn from events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, super-storm Sandy in the U.S.A. in 2012, and news articles spanning a large number of years and 100s of different disasters."

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
After studying literature about NLP it was obvious, that MLP is not the best neural network for the task. LSTM or RNN are preferred because they naturally have the capacity to store information about previously processed information and therefore can capture context of the message.
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

###### To deal with imbalance following measures were applied:
1. To validate the performance of ML pipeline following scoring metrics were applied:
    * precision (true positives / (true positives + false positives) )
    * recall (true positives  / (true positives + false negatives) )
    * f1 score (2 * (precision * recall) / (precision + recall) )
    * area under the reciever operating characteristic curve (ROC AUC) https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
2. To ensure that the influence of minority classes or categories is not shown in metrics macro averaging was applied to calculate composite score for every category as well as the overall score for a classifier. Macro averaging assigns equal weights to all classes so that if the score of a minority class has same impact as that of the majority class.
3. For MLP classifier upsampling of minority class was applied on training data.

### 5. Further thoughts on disaster response pipeline

A real disaster response pipeline has to deal with millions of messages. The fraction of related messages is much lower than in the presented dataset. For this scenario the following approach can be of advantage:
1. Create a classifier, which filters messages related to current disaster. This classifier must use additional data such as IP address or device location if possible. For training the NLP part of such a classifier a balanced dataset of related and unrelated messages is needed.
2. Create a classifier for coarse categories such as food, water, weather, earthquake etc.
3. Create a sub-categories classifier.
4. Extract proper nouns from messages to indicate names, addresses or landmarks.

### 6. Final pipeline and full classification report
#### 6.1. Final pipeline
CountVectorizer(analyzer='word', binary=False, decode_error='strict',
        dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 2), preprocessor=None, stop_words='english',
        strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
        tokenizer=<function tokenize at 0x7f0f012a2620>, vocabulary=None)
        
TfidfTransformer(use_idf = False)
        
MultiOutputClassifier(estimator=LinearSVC(C=1.0, class_weight='balanced', dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=1, tol=0.0001,
     verbose=0),
           n_jobs=1)
#### 6.2. Classification reports for all categories

Classification result for category "related":

             precision    recall  f1-score   support

          0       0.64      0.58      0.61      1540
          1       0.87      0.90      0.89      4967

    avg / total   0.82      0.82      0.82      6507

Classification result for category "request":

             precision    recall  f1-score   support

          0       0.94      0.91      0.93      5387
          1       0.63      0.71      0.67      1120

    avg / total   0.89      0.88      0.88      6507

Classification result for category "offer":

             precision    recall  f1-score   support

          0       1.00      1.00      1.00      6484
          1       0.50      0.04      0.08        23

    avg / total   0.99      1.00      0.99      6507

Classification result for category "aid_related":

             precision    recall  f1-score   support

          0       0.83      0.74      0.78      3828
          1       0.67      0.78      0.72      2679

    avg / total   0.76      0.75      0.76      6507

Classification result for category "medical_help":

             precision    recall  f1-score   support

          0       0.95      0.95      0.95      5988
          1       0.46      0.47      0.47       519

    avg / total   0.91      0.91      0.91      6507

Classification result for category "medical_products":

             precision    recall  f1-score   support

          0       0.97      0.98      0.97      6186
          1       0.50      0.49      0.50       321

    avg / total   0.95      0.95      0.95      6507

Classification result for category "search_and_rescue":

             precision    recall  f1-score   support

          0       0.98      0.99      0.99      6354
          1       0.46      0.22      0.30       153

    avg / total   0.97      0.98      0.97      6507

Classification result for category "security":

             precision    recall  f1-score   support

          0       0.98      1.00      0.99      6383
          1       0.23      0.05      0.08       124

    avg / total   0.97      0.98      0.97      6507

Classification result for category "military":

             precision    recall  f1-score   support

          0       0.98      0.98      0.98      6284
          1       0.51      0.55      0.53       223

    avg / total   0.97      0.97      0.97      6507

Classification result for category "water":

             precision    recall  f1-score   support

          0       0.99      0.97      0.98      6091
          1       0.66      0.81      0.72       416

    avg / total   0.97      0.96      0.96      6507

Classification result for category "food":

             precision    recall  f1-score   support

          0       0.98      0.96      0.97      5783
          1       0.74      0.81      0.78       724

    avg / total   0.95      0.95      0.95      6507

Classification result for category "shelter":

             precision    recall  f1-score   support

          0       0.97      0.95      0.96      5928
          1       0.60      0.71      0.65       579

    avg / total   0.94      0.93      0.93      6507

Classification result for category "clothing":

             precision    recall  f1-score   support

          0       0.99      1.00      0.99      6410
          1       0.65      0.55      0.60        97

    avg / total   0.99      0.99      0.99      6507

Classification result for category "money":

             precision    recall  f1-score   support

          0       0.99      0.99      0.99      6366
          1       0.42      0.45      0.44       141

    avg / total   0.98      0.97      0.97      6507

Classification result for category "missing_people":

             precision    recall  f1-score   support

          0       0.99      1.00      0.99      6425
          1       0.45      0.18      0.26        82

    avg / total   0.98      0.99      0.98      6507

Classification result for category "refugees":

             precision    recall  f1-score   support

          0       0.98      0.99      0.98      6266
          1       0.53      0.35      0.42       241

    avg / total   0.96      0.96      0.96      6507

Classification result for category "death":

             precision    recall  f1-score   support

          0       0.98      0.99      0.98      6214
          1       0.68      0.66      0.67       293

    avg / total   0.97      0.97      0.97      6507

Classification result for category "other_aid":

             precision    recall  f1-score   support

          0       0.91      0.89      0.90      5642
          1       0.38      0.44      0.41       865

    avg / total   0.84      0.83      0.84      6507

Classification result for category "infrastructure_related":

             precision    recall  f1-score   support

          0       0.95      0.96      0.95      6073
          1       0.28      0.23      0.25       434

    avg / total   0.90      0.91      0.90      6507

Classification result for category "transport":

             precision    recall  f1-score   support

          0       0.97      0.98      0.98      6208
          1       0.47      0.31      0.38       299

    avg / total   0.94      0.95      0.95      6507

Classification result for category "buildings":

             precision    recall  f1-score   support

          0       0.98      0.97      0.97      6169
          1       0.49      0.58      0.53       338

    avg / total   0.95      0.95      0.95      6507

Classification result for category "electricity":

             precision    recall  f1-score   support

          0       0.99      0.99      0.99      6379
          1       0.52      0.55      0.53       128

    avg / total   0.98      0.98      0.98      6507

Classification result for category "tools":

             precision    recall  f1-score   support

          0       0.99      1.00      1.00      6470
          1       0.00      0.00      0.00        37

    avg / total   0.99      0.99      0.99      6507

Classification result for category "hospitals":

             precision    recall  f1-score   support

          0       0.99      0.99      0.99      6447
          1       0.28      0.25      0.27        60

    avg / total   0.99      0.99      0.99      6507

Classification result for category "shops":

             precision    recall  f1-score   support

          0       0.99      1.00      1.00      6474
          1       0.00      0.00      0.00        33
          
    avg / total   0.99      0.99      0.99      6507

Classification result for category "aid_centers":

             precision    recall  f1-score   support

          0       0.99      1.00      0.99      6431
          1       0.28      0.11      0.15        76
          
    avg / total   0.98      0.99      0.98      6507

Classification result for category "other_infrastructure":

             precision    recall  f1-score   support

          0       0.96      0.98      0.97      6208
          1       0.25      0.17      0.20       299
          
    avg / total   0.93      0.94      0.93      6507

Classification result for category "weather_related":

             precision    recall  f1-score   support

          0       0.92      0.90      0.91      4657
          1       0.75      0.80      0.77      1850
          
    avg / total   0.87      0.87      0.87      6507

Classification result for category "floods":

             precision    recall  f1-score   support

          0       0.97      0.98      0.97      5975
          1       0.70      0.63      0.66       532
          
    avg / total   0.95      0.95      0.95      6507

Classification result for category "storm":

             precision    recall  f1-score   support

          0       0.98      0.96      0.97      5880
          1       0.69      0.78      0.73       627
    avg / total   0.95      0.94      0.95      6507

Classification result for category "fire":

             precision    recall  f1-score   support

          0       0.99      1.00      0.99      6448
          1       0.31      0.08      0.13        59
          
    avg / total   0.99      0.99      0.99      6507

Classification result for category "earthquake":

             precision    recall  f1-score   support

          0       0.98      0.98      0.98      5853
          1       0.85      0.82      0.84       654
          
    avg / total   0.97      0.97      0.97      6507

Classification result for category "cold":

             precision    recall  f1-score   support

          0       0.99      0.99      0.99      6369
          1       0.55      0.41      0.47       138
          
    avg / total   0.98      0.98      0.98      6507

Classification result for category "other_weather":

             precision    recall  f1-score   support

          0       0.96      0.98      0.97      6162
          1       0.42      0.33      0.37       345
          
    avg / total   0.93      0.94      0.94      6507

Classification result for category "direct_report":

             precision    recall  f1-score   support

          0       0.91      0.88      0.90      5216
          1       0.58      0.65      0.61      1291
          
    avg / total   0.85      0.84      0.84      6507
