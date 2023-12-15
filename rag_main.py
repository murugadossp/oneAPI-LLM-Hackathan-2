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

    def __init__(self):
        # Create a text generation pipeline

        self.llm_optimized = None
        self.qa_chain_optimized = None
        self.pipe_optimized = None
        self.qa_chain_not_optimized = None
        self.llm_not_optimized = None
        self.pipe_not_optimized = None
        self.chat_bot_model_optimized = ChatBotModel(optimize=True)
        self.chat_bot_model_not_optimized = ChatBotModel(optimize=False)

        embeddings = SentenceTransformerEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
        self.db = Chroma(persist_directory=PERSISTENT_DIR_PATH, embedding_function=embeddings)

        self.conversation_history = ""

    def rag_not_optimized(self, max_length=512,
                          temperature=0.3,
                          top_p=0.95):
        self.pipe_not_optimized = pipeline(
            'text2text-generation',
            model=self.chat_bot_model_not_optimized.model,
            tokenizer=self.chat_bot_model_not_optimized.tokenizer,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p
        )
        # Initialize a local language model pipeline
        self.llm_not_optimized = HuggingFacePipeline(pipeline=self.pipe_not_optimized)
        # Create a RetrievalQA chain
        self.qa_chain_not_optimized = RetrievalQA.from_chain_type(
            llm=self.llm_not_optimized,
            chain_type='stuff',
            retriever=self.db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
            return_source_documents=True,
        )

    def rag_optimized(self, max_length=512,
                      temperature=0.3,
                      top_p=0.95):
        self.pipe_optimized = pipeline(
            'text2text-generation',
            model=self.chat_bot_model_optimized.model,
            tokenizer=self.chat_bot_model_optimized.tokenizer,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p
        )
        # Initialize a local language model pipeline
        self.llm_optimized = HuggingFacePipeline(pipeline=self.pipe_optimized)
        # Create a RetrievalQA chain
        self.qa_chain_optimized = RetrievalQA.from_chain_type(
            llm=self.llm_not_optimized,
            chain_type='stuff',
            retriever=self.db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
            return_source_documents=True,
        )

    def generate_response_optimized(self, input_text):
        start_time = time.time()
        llm_response = self.qa_chain_optimized({"query": input_text})
        end_time = time.time()
        inference_time = end_time - start_time
        return llm_response['result'], round(inference_time,2)

    def generate_response_not_optimized(self, input_text):
        start_time = time.time()
        llm_response = self.qa_chain_not_optimized({"query": input_text})
        end_time = time.time()
        inference_time = end_time - start_time
        return llm_response['result'], round(inference_time,2)

    def front_end(self, input_text):
        if "exit" in input_text.lower():
            return f"Summary so far: \n\n {self.chat_logs} \n\n"

        # Assuming you have a variable to store conversation history
        if not hasattr(self, 'conversation_history'):
            self.conversation_history = ""

        # Add user input to conversation history
        self.conversation_history += f"You: {input_text}\n\n"

        resp_ipex, inference_ipex = self.generate_response_optimized(input_text)  # Your function to generate response
        resp_no_ipex, inference_no_ipex = self.generate_response_optimized(input_text)  # Your function to generate response
        self.chat_logs.append(
            {
                "query": input_text,
                "inference_ipex": inference_ipex,
                "inference_no_ipex": inference_no_ipex,
                "response_ipex": resp_ipex,
                "response_no_ipex": resp_no_ipex,

            }
        )
        # Add response to conversation history
        self.conversation_history += (f"NextGenAI Law Bot (ipex): {resp_ipex}\n\n "
                                      f"inference_time (ipex)   : {inference_ipex} seconds\n\n"
                                      f"NextGenAI Law Bot (No-ipex): {resp_no_ipex}\n\n "
                                      f"inference_time (No-ipex)   : {inference_no_ipex} seconds\n\n"
                                     )

        # Combine response and inference time
        display_text = f"{self.conversation_history}"

        return display_text

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

    chat_bot = ChatBot()
    chat_bot.launch_gradio_interface()
    # chat_bot.front_end_local()
