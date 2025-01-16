# Warning control
import warnings
import os
warnings.filterwarnings('ignore')
import signal
import logging
from dotenv import load_dotenv
import re
from langchain_aws import ChatBedrock
import boto3
import json
from pathlib import Path
from langchain_core.documents import Document
from uuid import uuid4
from PIL import Image as PILImage
import json
from langchain_community.embeddings import BedrockEmbeddings

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.vectorstores.utils import filter_complex_metadata

import numpy as np
import openai
import backoff

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph.message import add_messages
import nltk
from chatbot import Chatbot
from headers import run_pip_installations
import streamlit as st
from docling_converter import DoclingFileLoader

#from supabase import create_client, Client


from dotenv import load_dotenv
import os

load_dotenv()


# Initialize Supabase client
'''SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def register_user(email, password):
    response = supabase.auth.sign_up(email=email, password=password)
    return response

def login_user(email, password):
    response = supabase.auth.sign_in(email=email, password=password)
    return response

def save_chat_history(user_id, chat_history):
    data = {
        "user_id": user_id,
        "chat_history": chat_history
    }
    response = supabase.table('chat_history').insert(data).execute()
    return response

def load_chat_history(user_id):
    response = supabase.table('chat_history').select('*').eq('user_id', user_id).execute()
    return response.data'''


os.environ['USER_AGENT'] = os.getenv("USER_AGENT")
class PreProcessor:


    def __init__(self, p_dir, emb, llm, out_path):
        try:
            # Attempt to load the tokenizer
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            # If not found, download it
            print("Downloading the 'punkt' tokenizer...")
            nltk.download('punkt')
        # Initialize a set to keep track of added IDs
        self.elements = []
        self.persist_directory = p_dir
    
        self.model_name = emb
        self.output_path = out_path

        self.bedrock_client = boto3.client("bedrock-runtime", region_name="eu-central-1")
        self.embeddings = BedrockEmbeddings(
                client=self.bedrock_client, model_id=emb)

        os.environ["LANGCHAIN_TRACING_V2"]=os.getenv("LANGCHAIN_TRACING_V2")
        os.environ["LANGCHAIN_ENDPOINT"]=os.getenv("LANGCHAIN_ENDPOINT")
        os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
        os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

        self.llm = ChatBedrock(client=self.bedrock_client, model_id=llm)
        

    def process_table_text(self, text):

        pattern = re.compile(r'(?P<key>[A-Z\s]+)\s*([\d/,\s]+|N/A)', re.IGNORECASE)
        
        # Initialize a dictionary to store the extracted data
        table_data = {}

        # Normalize whitespace for easier parsing
        text = text.replace("\n", " ")  
        
        # Find all matches in the text using the defined pattern
        matches = pattern.findall(text)
        
        # Populate the table_data dictionary with extracted matches
        for match in matches:
            key = match[0].strip().replace(" ", "_").lower()  # Normalize the key
            value = match[1].strip() if match[1] else "N/A"  # Default to "N/A" if empty
            table_data[key] = value
        
        # Create a formatted summary of the available data
        formatted_table = "\n".join(
            [f"{key.replace('_', ' ').title()}: {value}" for key, value in table_data.items()]
        )

        return formatted_table



    def extract_table_data(self, table_element):
        """
        Extracts data from a table element, handling both structured tables and unstructured text.
        """
        try:
            # First, check if the table has rows and cells
            table_data = []
            if hasattr(table_element, "rows"):
                # Process as a structured table
                for row in table_element.rows:
                    row_data = [cell.text for cell in row.cells]
                    table_data.append(row_data)
            
            else:
                # Process as unstructured text using keywords
                text = table_element.text
                table_data = self.process_table_text(text)  # Parse the plain text data as per the process_table_text function
            print(table_data)
            return table_data
        
        except AttributeError as e:
            print(f"Error processing table element: {e}")
            return None

    # Function to generate embeddings using Amazon Titan Text Embeddings V2
    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=5)
    def generate_titan_embedding(self, input_text):
        # Create a Bedrock Runtime client in the AWS Region of your choice.
        

        # Set the model ID for Titan Text Embeddings V2.
        model_id = "amazon.titan-embed-text-v2:0"

        # Prepare the request for the Titan model.
        native_request = {"inputText": input_text}
        request = json.dumps(native_request)

        # Invoke the Titan model to get the embedding.
        response = self.client.invoke_model(modelId=model_id, body=request)

        # Decode the response and extract the embedding
        model_response = json.loads(response["body"].read())
        embedding = model_response["embedding"]
        
        return embedding

    # Backoff for embedding generation
    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=10)
    def generate_embedding(self, chunks, persist_dir=None):
        """Generate embedding for the user query with rate limit handling."""
        # Save the vector store
        if persist_dir:
            vector_store = Chroma.from_documents(
            documents=chunks,
            collection_name="chroma_index",
            embedding=self.embeddings,
            persist_directory=persist_dir,  # Where to save data locally, remove if not necessary
            )
        else:
            vector_store = Chroma.from_documents(
            documents=chunks,
            collection_name="chroma_index",
            embedding=self.embeddings,
            persist_directory=self.persist_directory,  # Where to save data locally, remove if not necessary
            )
        #docs = filter_complex_metadata(chunks)
        #vector_store.add_documents(documents=docs, ids=uuids)

        return vector_store


    
    # Function to initialize or load vector store
    def load_or_initialize_vector_store(self):
        try:
            # Attempt to load an existing vector store
            vector_store = Chroma(collection_name='chroma_index', persist_directory=self.persist_directory, embedding_function=self.embeddings)  # Using Chroma, replace with FAISS if necessary

            if vector_store:
                return vector_store

            else:
                print("No vector store found, initializing a new one.")
                chunks = self.process_pptx_data()

                # Initialize a new vector store
                # Save the new vector store

                vector_store = self.generate_embedding(chunks)
                return vector_store   # Return the new vector store

        except Exception as e:
            print(f"Error loading vector store: {e}")
            # If there's an error, create a new vector store from the provided chunks
            chunks = self.process_pptx_data()

            vector_store = self.generate_embedding(chunks)
            return vector_store  # Return the new vector store

    def get_files_from_directory(self, file_path):
        # If the input is a list of file paths, use it directly
        if isinstance(file_path, list):
            self._file_paths = file_path
        else:
            # If it's a directory path, check if it's a valid directory
            directory_path = Path(file_path)
            
            if directory_path.is_dir():
                # List all files (excluding directories) in the specified directory
                self._file_paths = [
                    str(directory_path / f) for f in os.listdir(directory_path)
                    if (directory_path / f).is_file()
                ]
            elif directory_path.is_file():
                # If it's a valid file, treat it as a single file path
                self._file_paths = [str(directory_path)]
            else:
                # If it's neither a file nor a directory, handle it accordingly
                print(f"The path {file_path} is neither a valid directory nor a file.")
                self._file_paths = []

    def process_pptx_data(self):
        self.get_files_from_directory(self.output_path)

        loader = DoclingFileLoader(self.bedrock_client, file_path=self._file_paths)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, add_start_index=True)
        docs = loader.load()
        print(docs)
        splits = text_splitter.split_documents(docs)
        
        return splits

    # Main function to tie everything together
    def process_directory(self):
        # Load or initialize the vector store
        #vector_store = self.load_or_initialize_vector_store()
        # Process the PPTX data again to obtain chunks
        chunks = self.process_pptx_data()

        # Handle empty chunks case
        if not chunks:
            print(f"No chunks created from the provided elements. Skipping...")
            return
        else:
            vector_store = self.generate_embedding(chunks, self.persist_directory) 
   
    def shutdown_app(self):
        """Function to shut down the Streamlit app."""
        # Get the current process id
        pid = os.getpid()
        # Send a termination signal to the current process
        os.kill(pid, signal.SIGINT)


    def delete_from_vectorstore(self, file_path):
        # Assume documents are indexed with a metadata field `file_path`
        try:
            file_directory, filename = os.path.split(file_path)
       
            vector_store = Chroma(
            collection_name="chroma_index",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,  # Where to save data locally, remove if not necessary
            )
            coll = vector_store.get()  # dict_keys(['ids', 'embeddings', 'documents', 'metadatas'])
            ids_to_del = []

            for idx in range(len(coll['ids'])):

                id = coll['ids'][idx]
                metadata = coll['metadatas'][idx]
                if metadata['filename'] == filename:
                    ids_to_del.append(id)

  
            vector_store.adelete(ids=ids_to_del) 
          
            print(f"Deleted vectorstore entry for {file_path}")

        except Exception as e:
            print(f"Error deleting from vectorstore: {e}")



if __name__ == "__main__":
    st.title("TERNA Chatbot")
    # Display chat history in the sidebar
    placeholder = st.empty()
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'context_history' not in st.session_state:
        st.session_state['context_history'] = []
    if not os.path.exists("./chroma_langchain_db"):
        os.mkdir("./chroma_langchain_db")
    
    processor = PreProcessor("./chroma_langchain_db", "amazon.titan-embed-text-v2:0", "anthropic.claude-3-5-sonnet-20240620-v1:0", os.path.join(os.getcwd(), 'files'))
    if not os.path.exists(processor.persist_directory) or len(os.listdir(processor.persist_directory)) <= 1:
        placeholder.write("Processing documents...")
       
        processor.process_directory()
        placeholder.empty()
        chatbot = Chatbot(processor)
        #have to separate it from the loading process
        chatbot.process_answer(st)

            
    else:
        chatbot = Chatbot(processor)
        #have to separate it from the loading process
        chatbot.process_answer(st)

    # Save chat history
    '''if st.button("Save Chat History"):
        if 'user_id' in st.session_state:
            user_id = st.session_state['user_id']
            save_chat_history(user_id, st.session_state['chat_history'])
            st.success("Chat history saved successfully!")
        else:
            st.error("User not logged in. Please log in to save chat history.")

if not os.path.exists('./files'):
    os.makedirs('./files')

st.title("TERNA Chatbot")

# User registration and login
st.sidebar.title("User Authentication")
auth_choice = st.sidebar.selectbox("Choose Authentication", ["Login", "Register"])

if auth_choice == "Register":
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Register"):
        response = register_user(email, password)
        st.sidebar.success("User registered successfully!")

if auth_choice == "Login":
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        response = login_user(email, password)
        if response.user:
            st.sidebar.success("Logged in successfully!")
            user_id = response.user.id
            st.session_state['user_id'] = user_id
            chat_history = load_chat_history(user_id)
            st.session_state['chat_history'] = chat_history
        else:
            st.sidebar.error("Login failed!")

if 'user_id' in st.session_state:
    user_id = st.session_state['user_id']
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'context_history' not in st.session_state:
        st.session_state['context_history'] = []

    processor = PreProcessor("./chroma_langchain_db", "amazon.titan-embed-text-v2:0", "eu.meta.llama3-2-1b-instruct-v1:0", "./unstructured-output/")
    if not os.path.exists(processor.persist_directory) or len(os.listdir(processor.persist_directory)) <= 1:
        placeholder = st.empty()
        placeholder.write("Processing documents...")
        processor.process_directory()
        placeholder.empty()
        if st.button("Process Documents"):
            placeholder.write("Processing documents...")
            processor.process_directory()
        placeholder.empty()
        if st.button("Clear Chat History"):
            st.session_state['chat_history'].clear()
            st.session_state['context_history'].clear()
        if st.button("Shut Down App"):
            st.warning("Shutting down the app...")
            processor.shutdown_app()
        chatbot = Chatbot(os.getcwd(), processor, query=None)
        chatbot.process_answer(st)
    else:
        if st.button("Process Documents"):
            placeholder.write("Processing documents...")
            processor.delete_directory_contents(processor.persist_directory)
            processor.process_directory()
        placeholder.empty()
        if st.button("Clear Chat History"):
            st.session_state['chat_history'].clear()
            st.session_state['context_history'].clear()
        if st.button("Shut Down App"):
            st.warning("Shutting down the app...")
            processor.shutdown_app()
        chatbot = Chatbot(os.getcwd(), processor, query=None)
        chatbot.process_answer(st)

    # Save chat history
    if st.button("Save Chat History"):
        save_chat_history(user_id, st.session_state['chat_history'])
        st.success("Chat history saved successfully!")

if not os.path.exists('./files'):
    os.makedirs('./files')'''