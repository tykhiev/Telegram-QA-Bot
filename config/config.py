import os
from redis import StrictRedis
import weaviate
from langchain.vectorstores.weaviate import Weaviate

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')
WEAVIATE_URL = os.getenv('WEAVIATE_URL')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')

REDIS_CLIENT = StrictRedis(host="localhost", port=6379, decode_responses=True)

WEAVIATE_CLIENT = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    additional_headers={
        "X-OpenAI-Api-Key": OPENAI_API_KEY
    }
)


def vector_db(docs, embeddings):
    VECTORDB = Weaviate.from_documents(
        docs, embeddings, weaviate_url=WEAVIATE_URL, by_text=False)
    return VECTORDB


DB_DIR = 'data/db'
OUTPUT_DIR = 'data/output'
