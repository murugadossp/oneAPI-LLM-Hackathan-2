# Import necessary modules and classes
import os
import time

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from chatbot_model import ChatBotModel
from loggers import *

PERSISTENT_DIR_PATH = "/home/sdp/vector_db/chroma_db_2"


class InputDataLoader:
    """
    InputDataLoader is a class for loading the PDF documents given in a directory

    Attributes:
    - persistent_db_dir: Directory Path for DB to store
    - chunk_size: chunk_size default 1200 bytes .
    - chunk_overlap : 200 bytes
    """

    def __init__(self,
                 persistent_db_dir=PERSISTENT_DIR_PATH,
                 chunk_size=1200,
                 chunk_overlap=200
                 ) -> None:
        # Define the directory where the Chroma
        # database will persist its data
        # Make sure the directory exists, create if it doesn't
        self.db = None
        self.embeddings = None
        self.texts = None
        self.persistent_db_dir = persistent_db_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if not os.path.exists(persistent_db_dir):
            info(f"creating persistent database directory: {persistent_db_dir}")
            os.makedirs(persistent_db_dir)

    def document_loader(self):
        # Initialize a directory loader to load PDF documents from a directory
        loader = DirectoryLoader("data",
                                 glob="./*.pdf",
                                 loader_cls=PyPDFLoader
                                 )
        documents = loader.load()

        # Initialize a text splitter to split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                       chunk_overlap=self.chunk_overlap)

        # Split the loaded documents into chunks
        self.texts = text_splitter.split_documents(documents)

    def core(self):
        start_time = time.time()
        # Get the Documents to texts
        self.document_loader()
        # Creating a Vector DB using Chroma DB and SentenceTransformerEmbeddings
        # Initialize SentenceTransformerEmbeddings with a pre-trained model
        info("Initializing SentenceTransformerEmbeddings")
        # self.embeddings = SentenceTransformerEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create a Chroma vector database from the text chunks
        info("Creating a Chroma vector database from the text chunks")
        self.db = Chroma.from_documents(self.texts,
                                        self.embeddings,
                                        persist_directory=self.persistent_db_dir)
        info(f"Making the Chroma vector database persistent")
        self.db.persist()
        info(f"Total Time taken for Converting the Documents to Embeddings and store in the database: "
             f"{round((time.time() - start_time),2)}")


if __name__ == '__main__':
    input_loader = InputDataLoader(persistent_db_dir= "/home/sdp/vector_db/chroma_db2")
    input_loader.core()


    # info(f"input_loader texts: {input_loader.texts}")
    # info(f"input_loader embeddings: {input_loader.embeddings}")
