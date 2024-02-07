import os
from redis import StrictRedis
import weaviate
from langchain.vectorstores.weaviate import Weaviate
import pinecone

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


class config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    DB_DIR = 'data/db'
    OUTPUT_DIR = 'data/output'
