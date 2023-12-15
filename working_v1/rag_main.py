# Import necessary modules and classes
import gradio as gr
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores.chroma import Chroma
from transformers import pipeline

from chatbot_model import ChatBotModel
from input_data_loader import InputDataLoader

from loggers import *

PERSISTENT_DIR_PATH = "/home/sdp/vector_db/chroma_db"

class ChatBot:
    def __init__(self, chat_bot_model, db):
        # Create a text generation pipeline
        self.pipe = pipeline(
            'text2text-generation',
            model=chat_bot_model.model,
            tokenizer=chat_bot_model.tokenizer,
            max_length=512,
            do_sample=True,
            temperature=0.3,
            top_p=0.95
        )
        # Initialize a local language model pipeline
        self.local_llm = HuggingFacePipeline(pipeline=self.pipe)

        # Create a RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.local_llm,
            chain_type='stuff',
            retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
            return_source_documents=True,
        )

    def front_end(self, input_query):
        # Removed the loop and input since Gradio handles this
        if input_query.upper() == 'EXIT':
            return "Exiting..."

        llm_response = self.qa_chain({"query": input_query})
        return llm_response['result']

    def front_end_local(self):
        while True:
            input_query = input("Enter your query (or 'EXIT' to stop): ")

            # Check if the user entered 'EXIT' to stop the program
            if input_query.upper() == 'EXIT':
                info("Exiting...")
                break

            # Execute the query using the QA chain
            info("Executing query using the QA chain")
            llm_response = self.qa_chain({"query": input_query})

            # Print the response
            info(llm_response['result'])

    def launch_gradio_interface(self):
        interface = gr.Interface(
            fn=self.front_end,
            inputs="text",
            outputs="text",
            title="ChatBot Interface",
            description="Enter your query below:"
        )
        interface.launch(server_port=7902)


if __name__ == '__main__':
    # Usage
    chat_bot_model = ChatBotModel()
    embeddings = SentenceTransformerEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
    db = Chroma(persist_directory=PERSISTENT_DIR_PATH, embedding_function=embeddings)
    chat_bot = ChatBot(chat_bot_model, db)
    chat_bot.launch_gradio_interface()
    # chat_bot.front_end_local()
