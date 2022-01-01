import sys
import pandas as pd
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges datasets from 2 filepaths.
    
    Arguments:
        messages_filepath = Path to the CSV file containing messages
        categories_filepath = Path to the CSV file containing categories
    Output:
        df = Merged data containing messages and categories
    
    """
    messages = pd.read_csv('disaster_messages.csv')
    categories = pd.read_csv('disaster_categories.csv')
    df = messages.merge(categories, how='inner', on=['id'])
    return df


def clean_data(df):
    """
    Cleans DataFrame
    
    Arguments:
        df = DataFrame
    Outputs:
        df = DataFrame after cleaned up
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.head(1)
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0,:]
    # rename the columns of `categories`
    categories.columns = category_colnames
    # Iterate through the category columns in df
    # to keep only the last character of each string (the 1 or 0)
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # replace 2 with 1 in related column since we only have (0,1)
    categories['related'] = categories['related'].replace(to_replace=2, value=1)
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Save the DataFrame in SQLite database.
    """
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('disaster_messages', engine, index=False) 


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