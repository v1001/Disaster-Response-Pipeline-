import sys
# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load data from specified files

    Keyword arguments:
    messages_filepath -- file path to the file with messages
    categories_filepath -- file path to the file with categories
    
    Return:
    pandas DataFrame with both messages and categories
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on = ['id'])
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df['categories'].str.split(';').tolist(), columns = df['categories'][0].split(';'))
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [x[:-2] for x in row.tolist()]
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    categories.head()
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories)
    return(df)

def clean_data(df):
    """Cleans data from duplicates, errors and empty columns

    Keyword arguments:
    df -- pandas DataFrame
    
    Return:
    pandas DataFrame
    """
    df = df.loc[:, (df != 0).any(axis=0)]
    df = df[df['related']!=2]
    return df

def save_data(df, database_filename):
    """Saves DataFrame to a SQL database

    Keyword arguments:
    df -- pandas DataFrame
    database_filename -- file path to the file with database
    
    Return:
    None
    """
    
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('disaster_tweets', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()