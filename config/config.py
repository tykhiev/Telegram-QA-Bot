import os

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')
WEAVIATE_URL = os.getenv('WEAVIATE_URL')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')

DB_DIR = 'data/db'
OUTPUT_DIR = 'data/output'
