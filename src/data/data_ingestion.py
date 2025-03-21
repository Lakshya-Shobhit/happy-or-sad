import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# logging configuration
logger = logging.getLogger('data_ingestin')
logger.setLevel('DEBUG')
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def load_params(params_path: str) -> dict:
    """ Loads configuration from params.yaml """
    logger.info('Entring load_params')
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            logger.info('Exiting load_params %s', params)
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


def load_data(data_path: str) -> pd.DataFrame:
    """ Loads data from given path """
    logger.info('Entring load_data')
    try:
        raw_data = pd.read_csv(data_path)
        logger.info('Data Loaded')
        return raw_data
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    logger.info('Entring preprocess_data')
    try:
        data.drop(columns=['tweet_id'], inplace=True)
        final_data = data[data['sentiment'].isin(['happiness', 'sadness'])]
        final_data['sentiment'].replace(
            {'happiness': 1, 'sadness': 0}, inplace=True
            )
        logger.debug('Data preprocessing completed')
        return final_data
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise


def save_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str
) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(
            os.path.join(raw_data_path, "train.csv"), index=False
            )
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main() -> None:
    """Main function to load, train-test split and save data"""
    URL = (
        "https://raw.githubusercontent.com/campusx-official/"
        "jupyter-masterclass/main/tweet_emotions.csv"
    )
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        data = load_data(data_path=URL)
        final_data = preprocess_data(data=data)
        train_data, test_data = train_test_split(
            final_data, test_size=test_size, random_state=42
            )
        save_data(train_data, test_data, './data')
    except Exception as e:
        logger.error('Unexpected error occurred during data ingestion: %s', e)
        raise


if __name__ == '__main__':
    main()
