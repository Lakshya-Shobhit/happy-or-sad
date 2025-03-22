import numpy as np
import pandas as pd
import pickle
import yaml
import logging
from sklearn.ensemble import GradientBoostingClassifier


# logging configuration
logger = logging.getLogger('model_building')
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
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def train_model(
    x_train: np.ndarray, y_train: np.ndarray, params: dict
) -> GradientBoostingClassifier:
    """Train a Gradient Boosting Classifier Model """
    logger.info('Entring train_model')
    try:
        clf = GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'])
        clf.fit(x_train, y_train)
        logger.info('Exiting train_model')
        return clf
    except Exception as e:
        logger.error(
            'Unexpected error occurred while training the model: %s', e)
        raise


def save_model(model, model_path: str) -> None:
    """ Save the trained model """
    logger.info('Entring save_model')
    logger.debug('Save model to path : %s', model_path)
    try:
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        logger.info('Exiting save_model')
    except Exception as e:
        logger.error('Unexpected error occurred while saving the model: %s', e)
        raise


def main() -> None:
    try:
        params = load_params('params.yaml')['model_building']
        train_data = load_data('./data/processed/train_bow.csv')
        x_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        clf = train_model(x_train, y_train, params)
        save_model(clf, './models/model.pkl')
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
