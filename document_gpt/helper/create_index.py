import os

from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.directory import DirectoryLoader
from langchain.document_loaders.text import TextLoader
from PyPDF2 import PdfReader

from config import config


def create_index(file_path: str) -> None:
    try:
        print(f'Processing file: {file_path}')
        reader = PdfReader(file_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text()

        with open(f'{config.OUTPUT_DIR}/output.txt', 'w') as file:
            file.write(text)

        print('Creating index...')

        loader = DirectoryLoader(
            config.OUTPUT_DIR,
            glob='**/*.txt',
            loader_cls=TextLoader
        )

        documents = loader.load()
        print(f'Loaded {len(documents)} documents.')
        text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=1024,
            chunk_overlap=128
        )
        print('Splitting documents...')

        texts = text_splitter.split_documents(documents)
        print(f'Split {len(texts)} documents.')

        embeddings = OpenAIEmbeddings(
            openai_api_key=config.OPENAI_API_KEY
        )
        print('Creating embeddings...')

        persist_directory = config.DB_DIR
        print('Creating vector store...')

        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print('Vector store created.')

        vectordb.persist()

    except Exception as e:
        # Print the exception for debugging purposes
        print(f"An error occurred: {e}")
