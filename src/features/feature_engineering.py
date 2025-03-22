import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import yaml
import logging


# logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def load_params(params_path: str) -> dict:
    """ Load parameters from a YAML file """
    logger.info('Entring Load _params')
    logger.debug('params_path: %s', params_path)
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.info('Exiting Load _params')
        return params
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """ Load data from a CSV file """
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def apply_bag_of_words(
    train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int
) -> tuple:
    """ Apply Bag of Words to the data """
    logger.info('Entring bag_of_words')
    try:
        vectorizer = CountVectorizer(max_features=max_features)

        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logger.info('Bag of Words applied and data transformed')
        return train_df, test_df
    except Exception as e:
        logger.error('Error during Bag of Words transformation: %s', e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """ Save the dataframe to a CSV file """
    logger.info('Entring save_data')
    logger.debug('fil_path %s', file_path)
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info('Exiting save data')
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    """ Save features """
    logger.info('Entring main')
    try:
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')
        # Aply Bag of Words
        train_df, test_df = apply_bag_of_words(
                                train_data, test_data, max_features
                                )
        # Save features in processed folder
        save_data(train_df, os.path.join(
            "./data", "processed", "train_bow.csv"))
        save_data(test_df, os.path.join(
            "./data", "processed", "test_bow.csv"))  
    except Exception as e:
        logger.error(
            'Failed to complete the feature engineering process: %s', e)
        raise


if __name__ == '__main__':
    main()
