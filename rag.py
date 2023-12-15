# Import necessary modules and classes
import os

from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import intel_extension_for_pytorch as ipex
from device import *
from loggers import *

MODEL_CACHE_PATH = "/home/common/data/Big_Data/GenAI/llm_models"
import torch

PERSISTENT_DIR_PATH = "/home/sdp/vector_db/chroma_db"


class InputDataLoader:
    """
    ChatBotModel is a class for Setting up the pretrained tokenizer and model.

    Attributes:
    - model_id_or_path: model Id for text generation. Default is ""MBZUAI/LaMini-Flan-T5-783M""
    - torch_dtype: The data type to use in the model.
    - optimize : If True Intel Optimizer for pytorch is used.
    """

    def __init__(self,
                 persistent_db_dir=PERSISTENT_DIR_PATH,
                 chunk_size=1200,
                 chunk_overlap=200
                 ) -> None:
        # Define the directory where the Chroma
        # database will persist its data
        # Make sure the directory exists, create if it doesn't
        self.texts = None
        self.persistent_db_dir = persistent_db_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if not os.path.exists(persistent_db_dir):
            info(f"creating persistent database directory: {persistent_db_dir}")
            os.makedirs(persistent_db_dir)

    def document_loader(self):
        # Initialize a directory loader to load PDF documents from a directory

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

   


if __name__ == '__main__':
    chat_bot_model = ChatBotModel()
    info(f"chat_bot_model tokenizer: {chat_bot_model.tokenizer}")
    info(f"chat_bot_model model: {chat_bot_model.model}")
