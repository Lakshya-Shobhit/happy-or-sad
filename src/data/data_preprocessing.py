import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import logging


# logging configuration
logger = logging.getLogger('data_preprcessing')
logger.setLevel('DEBUG')
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Download Words
nltk.download('wordnet')
nltk.download('stopwords')


def lematization(text: str) -> str:
    """ reducing a word to its base """
    logger.info('Entring Lematization')
    lematizer = WordNetLemmatizer()
    text = text.split()
    text = [lematizer.lemmatize(word) for word in text]
    logger.info('Exit Lematization')
    return " ".join(text)


def remove_stop_words(text: str) -> str:
    """ removing stop words from the text """
    logger.info('Entring remove_stop_words')
    stopword = set(stopwords.words('english'))
    Text = [i for i in str(text).split() if i not in stopword]
    logger.info('Exiting remove_stop_words')
    return " ".join(Text)


def removing_numbers(text: str) -> str:
    """ Remove number from test """
    text = text.split()
    text = [char for char in text if not char.isdigit()]
    return " ".join(text)


def removing_punctuations(text: str) -> str:
    """ Remove punctuations """
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def removing_urls(text: str) -> str:
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def lower_case(text):
    return text.lower()


def normalize_text(df):
    """ Normalize text """
    logger.info('Entring Normaize Text')
    df.content = df.content.apply(lambda content: lower_case(content))
    df.content = df.content.apply(lambda content: remove_stop_words(content))
    df.content = df.content.apply(lambda content: removing_numbers(content))
    df.content = df.content.apply(
        lambda content: removing_punctuations(content)
        )
    df.content = df.content.apply(lambda content: removing_urls(content))
    df.content = df.content.apply(lambda content: lematization(content))
    logger.info('Exiting Normalize Text')
    return df


def main() -> None:
    """ Preprocess data """
    logger.info('Entring main')
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)
        train_processed_data.fillna('', inplace=True)
        test_processed_data.fillna('', inplace=True)
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        train_processed_data.to_csv(
            os.path.join(data_path, "train_processed.csv")
            )
        test_processed_data.to_csv(
            os.path.join(data_path, "test_processed.csv")
            )
        logger.info('Exiting main')
    except Exception as e:
        logger.error("Error while prepprosseccing: %s", e)


if __name__ == '__main__':
    main()
