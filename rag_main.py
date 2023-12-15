# Import necessary modules and classes
import time

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
    chat_logs = []

    def __init__(self, chat_bot_model, db):
        # Create a text generation pipeline
        self.conversation_history = ""
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

    def front_end(self, input_text):
        if "exit" in input_text.lower():
            return f"Summary so far: \n\n {self.chat_logs} \n\n"

        # Assuming you have a variable to store conversation history
        if not hasattr(self, 'conversation_history'):
            self.conversation_history = ""

        # Add user input to conversation history
        self.conversation_history += f"You: {input_text}\n\n"

        # Measure inference time
        start_time = time.time()
        response = self.generate_response(input_text)  # Your function to generate response
        end_time = time.time()
        inference_time = end_time - start_time
        self.chat_logs.append(
            {
                "query": input_text,
                "response": response,
                "inference_time": inference_time,
                "optimization": True
            }
        )
        # Add response to conversation history
        self.conversation_history += (f"NextGenAI Law Bot: {response}\n\n "
                                      f"inference_time: {inference_time}"
                                      f"\n\n")

        # Combine response and inference time
        display_text = f"{self.conversation_history}\n(Inference time: {inference_time:.2f} seconds)"

        return display_text

    def generate_response(self, input_text):
        llm_response = self.qa_chain({"query": input_text})
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
            title="NextGenAI ChatBot",
            description="Enter your query about Indian Constitution and Sections of the law"
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
