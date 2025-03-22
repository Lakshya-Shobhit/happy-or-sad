import numpy as np
import pandas as pd
import pickle
import json
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score)

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


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


def load_model(file_path: str):
    """Load the trained model """
    logger.info('Entring into load model')
    logger.debug('file_path : %s', file_path)
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.info('Exiting load model')
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error(
            'Unexpected error occurred while loading the model: %s', e)
        raise


def evaluate_model(clf, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    """ Evaluate the model and return evaluation metrices """
    logger.info('Entring evaluation model')
    try:
        y_pred = clf.predict(x_test)
        y_pred_proba = clf.predict_proba(x_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }

        logger.info(
            'Model evaluation metrics calculated metrices %s', metrics_dict)
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    logger.info('Entring into save_metrics')
    logger.debug('')
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise


def main():
    try:
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_bow.csv')

        x_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, x_test, y_test)

        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        raise


if __name__ == '__main__':
    main()
