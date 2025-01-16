import os
import signal
import base64
import io
import subprocess
import getpass
from uuid import uuid4
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import streamlit as st
import boto3
import nltk
from langchain_aws import ChatBedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_core.documents import Document
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured_ingest.connector.local import SimpleLocalConfig
from unstructured_ingest.interfaces import PartitionConfig, ProcessorConfig, ReadConfig
from unstructured_ingest.runner import LocalRunner
from docling_converter import DoclingFileLoader
from supabase import create_client, Client
from pathlib import Path
import re

load_dotenv()

# Now you can access the environment variables
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class PreProcessor:
    def __init__(self, p_dir, emb, llm, out_path):
        self.elements = []
        self.persist_directory = p_dir

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading the 'punkt' tokenizer...")
            nltk.download('punkt')

        self.model_name = emb
        self.output_path = out_path
        self.bedrock_client = boto3.client("bedrock-runtime", region_name="eu-central-1")
        self.embeddings = BedrockEmbeddings(client=self.bedrock_client, model_id=emb)

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = "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"
        os.environ["LANGCHAIN_PROJECT"] = "TERNA-chatbot"
        os.environ["OPENAI_API_KEY"] = "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"

        self.llm = ChatBedrock(client=self.bedrock_client, model_id=llm)

    def huggingface_login(self):
        hf_token = getpass.getpass("Enter your Hugging Face token: ")
        try:
            subprocess.run(f"echo {hf_token} | huggingface-cli login", shell=False, check=True)
            print("Successfully logged into Hugging Face!")
        except subprocess.CalledProcessError as e:
            print(f"Error during Hugging Face login: {e}")

    def ingest_documents(self, directory_path):
        os.makedirs(self.output_path, exist_ok=True)
        runner = LocalRunner(
            processor_config=ProcessorConfig(
                verbose=False,
                output_dir=self.output_path,
                num_processes=2,
            ),
            read_config=ReadConfig(),
            partition_config=PartitionConfig(
                partition_by_api=False,
                api_key=os.getenv("UNSTRUCTURED_API_KEY"),
                strategy="hi_res",
            ),
            connector_config=SimpleLocalConfig(input_path=directory_path, recursive=False),
        )
        runner.run()
        print("Document ingestion completed. Output saved in:", self.output_path)
        return self.output_path

    def get_image_block_types(self, file_elements, docs):
        tables = []
        for element in file_elements:
            if element.category != "Table":
                metadata = element.metadata
                if "image_base64" in metadata or element.category == "Image":
                    image_data = base64.b64decode(metadata["image_base64"])
                    image = Image.open(io.BytesIO(image_data))
                    text_from_image = pytesseract.image_to_string(image)

                    doc = Document(
                        page_content=text_from_image,
                        metadata=metadata,
                        id=str(uuid4())
                    )
                    docs.append(doc)
                if hasattr(metadata, "to_dict"):
                    metadata = metadata.to_dict()
                elif not isinstance(metadata, dict):
                    continue
            else:
                tables.append(element)

        for table in tables:
            page_content = table.text
            if hasattr(table.metadata, 'to_dict'):
                metadata = table.metadata.to_dict()
            else:
                metadata = {
                    'source': 'unknown',
                    'content': table.metadata.text_as_html
                }
            doc = Document(
                page_content=page_content,
                metadata=metadata,
                id=str(uuid4())
            )
            docs.append(doc)
        return docs

    def process_table_text(self, text):
        pattern = re.compile(r'(?P<key>[A-Z\s]+)\s*([\d/,\s]+|N/A)', re.IGNORECASE)
        table_data = {}
        text = text.replace("\n", " ")
        matches = pattern.findall(text)
        for match in matches:
            key = match[0].strip().replace(" ", "_").lower()
            value = match[1].strip() if match[1] else "N/A"
            table_data[key] = value
        formatted_table = "\n".join(
            [f"{key.replace('_', ' ').title()}: {value}" for key, value in table_data.items()]
        )
        return formatted_table

    def generate_embedding(self, chunks):
        """Generate embedding for the user query with rate limit handling."""
        uuids = [str(uuid4()) for _ in range(len(chunks))]

        # Save the vector store
        vector_store = Chroma(
            collection_name="chroma_index",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,  
        )
        docs = filter_complex_metadata(chunks)
        vector_store.add_documents(documents=docs, ids=uuids)

        return vector_store

    def load_or_initialize_vector_store(self, embeddings, elements):
        try:
            response = supabase.table("documents").select("*").execute()
            if response.data:
                return supabase
            else:
                print("No vector store found, initializing a new one.")
                chunks = self.process_pptx_data(elements)
                vector_store = self.generate_embedding(chunks)
                return vector_store
        except Exception as e:
            print(f"Error loading vector store: {e}")
            chunks = self.process_pptx_data(elements)
            vector_store = self.generate_embedding(chunks)
            return vector_store

    def get_files_from_directory(self, file_path):
        if isinstance(file_path, list):
            self._file_paths = file_path
        else:
            directory_path = Path(file_path)
            if directory_path.is_dir():
                self._file_paths = [
                    str(directory_path / f) for f in os.listdir(directory_path)
                    if (directory_path / f).is_file()
                ]
            elif directory_path.is_file():
                self._file_paths = [str(directory_path)]
            else:
                raise ValueError(f"The path {file_path} is neither a valid directory nor a file.")

    def process_pptx_data(self, pptx_elements=None):
        file_list = self.get_files_from_directory(os.path.join(os.getcwd(), 'files'))
        loader = DoclingFileLoader(file_path=self._file_paths)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, add_start_index=True)
        docs = loader.load()
        splits = text_splitter.split_documents(docs)
        return splits

    def process_directory(self, elements, query=None, max_tokens=1000):
        vector_store = self.load_or_initialize_vector_store(self.embeddings, elements)
        chunks = self.process_pptx_data(elements)
        if not chunks:
            print(f"No chunks created from the provided elements. Skipping...")
            return
        if vector_store:
            uuids = [str(uuid4()) for _ in range(len(chunks))]
            if not uuids:
                print(f"No UUIDs generated for chunks. Skipping...")
                return
            docs = filter_complex_metadata(chunks)
            vector_store.add_documents(documents=docs, ids=uuids)
        else:
            print("Error: Vector Store not found! Creating and loading...")
            vector_store = self.generate_embedding(chunks)

    def shutdown_app(self):
        pid = os.getpid()
        os.kill(pid, signal.SIGINT)

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
    return response.data

st.title("TERNA Chatbot")

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
        placeholder = st.empty()
        placeholder.write("Processing documents...")
        processor.process_directory()
        placeholder.empty()
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

    if st.button("Save Chat History"):
        save_chat_history(user_id, st.session_state['chat_history'])
        st.success("Chat history saved successfully!")

if not os.path.exists('./files'):
    os.makedirs('./files')
