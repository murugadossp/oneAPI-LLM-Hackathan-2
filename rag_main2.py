# Import necessary modules and classes
import time

import gradio as gr
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough

from chatbot_model import ChatBotModel

from loggers import *

PERSISTENT_DIR_PATH = "/home/sdp/vector_db/chroma_db2"


class ChatBot:
    chat_logs = []

    def __init__(self):
        # Create a text generation pipeline

        self.rag_chain_not_optimized = None
        self.rag_chain_optimized = None
        self.llm_optimized = None
        self.qa_chain_optimized = None
        self.pipe_optimized = None
        self.qa_chain_not_optimized = None
        self.llm_not_optimized = None
        self.pipe_not_optimized = None
        self.chat_bot_model_optimized = ChatBotModel(
                 model_id_or_path = "mistralai/Mixtral-8x7B-Instruct-v0.1",
                 optimize=True
                    )
        self.chat_bot_model_not_optimized = ChatBotModel(
                 model_id_or_path = "mistralai/Mixtral-8x7B-Instruct-v0.1",
                 optimize=False
                    )

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = Chroma(persist_directory=PERSISTENT_DIR_PATH, embedding_function=embeddings)
        self.llm_optimized = self._pipeline_mistral(bot_model=self.chat_bot_model_optimized)
        self.llm_not_optimized = self._pipeline_mistral(bot_model=self.chat_bot_model_not_optimized)
        self.rag_chain()
        self.conversation_history = ""


    def _pipeline_mistral(self, max_length=512, temperature=0.3, top_p=0.95, bot_model=None):
        text_generation_pipeline = pipeline(
            model=bot_model.model,
            tokenizer=bot_model.tokenizer,
            task="text-generation",
            temperature=temperature,
            do_sample=True,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=1000,
        )
        return HuggingFacePipeline(pipeline=text_generation_pipeline)

    def prompt_template(self):
        prompt_template = """
        ### [INST] 
        Instruction: Assume You are expert in India's constitution and you know about India Law Sections, which is otherwise Indian Penal Code. 
        Here is context to help:

        {context}

        ### QUESTION:
        {question} 

        [/INST]
         """
        # Create prompt from prompt template
        self.prompt = PromptTemplate(
                            input_variables=["context", "question"],
                            template=prompt_template,
                            )

    def rag_chain(self):
        # retriever = db.as_retriever()
        retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        self.prompt_template()
        # Create llm chain
        llm_chain_optimized = LLMChain(llm=self.llm_optimized, prompt=self.prompt)
        # Create llm chain
        llm_chain_not_optimized = LLMChain(llm=self.llm_not_optimized, prompt=self.prompt)

        self.rag_chain_not_optimized = (
                {"context": retriever, "question": RunnablePassthrough()}
                | llm_chain_not_optimized
        )
        self.rag_chain_optimized = (
                {"context": retriever, "question": RunnablePassthrough()}
                | llm_chain_optimized
        )


    def generate_response(self, input_query, use_optimized_chain=True):
        start_time = time.time()

        # Decide which chain to use based on the input parameter
        if use_optimized_chain:
            llm_response = self.rag_chain_optimized.invoke(input_query)
        else:
            llm_response = self.rag_chain_not_optimized.invoke(input_query)

        inference_time = round((time.time() - start_time), 2)

        # Print the response
        print(f"llm_response: {llm_response['text']}")
        print(f"inference_time: {inference_time}")

        return llm_response['text'], inference_time

    def front_end(self, input_text):
        if "exit" in input_text.lower():
            return f"Summary so far: \n\n {self.chat_logs} \n\n"

        # Assuming you have a variable to store conversation history
        if not hasattr(self, 'conversation_history'):
            self.conversation_history = ""

        # Add user input to conversation history
        self.conversation_history += f"You: {input_text}\n\n"

        resp_ipex, inference_ipex = self.generate_response(input_text, True)
        resp_no_ipex, inference_no_ipex = self.generate_response(input_text, False)
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
            info("Executing query")
            self.front_end(input_query)

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
    # chat_bot.launch_gradio_interface()
    chat_bot.front_end_local()
