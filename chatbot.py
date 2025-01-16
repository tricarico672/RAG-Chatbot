# Warning control
import warnings
warnings.filterwarnings('ignore')
import time
import random
from typing import Iterable
from langchain_core.documents import Document
from unstructured.documents.elements import Image
from uuid import uuid4
from PIL import Image as PILImage
from langdetect import detect
from deep_translator import GoogleTranslator
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import re
from unstructured.staging.base import elements_from_json

import numpy as np
import openai
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
#

from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from dotenv import load_dotenv
from transformers import GPT2TokenizerFast



class Chatbot:


    def __init__(self, processor):
        self.processor = processor


    # Simulate a function to get a streaming response from a language model
    def stream_response(self, response):
        for i in range(1, 6):  # Simulate response in 5 chunks
            time.sleep(1)  # Simulate time delay for streaming
            yield response
    def is_greeting(self, query):
        # Regex pattern for detecting greetings
        greeting_pattern = r'\b(hello|hi|hey|howdy|ciao|buon|giorno|sera|buona|greetings|' \
                        r'good (morning|afternoon|evening)|' \
                        r'what(\'s| is) up|yo|sup)\b'
        return bool(re.search(greeting_pattern, query.lower()))
    def is_irrelevant(self, answer):
        pattern = r"^(I\s*apologize|Mi\s*scuso|Non\s*posso|^Non\s*ho\s*sufficienti\s*informazioni\s*per\s*fornire)"
        
        # Check if the pattern matches the response
        return bool(re.search(pattern, answer, re.IGNORECASE))

    # Main QA pipeline
    def qa_pipeline(self, st, vectorstore, question, results=None):

        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        if 'context_history' not in st.session_state:
            st.session_state['context_history'] = []
        index = 0
        # Step 1: Display speakers and messages
        for idx, (speaker, message) in enumerate(st.session_state.chat_history):
            st.write(f"**{speaker}:** {message}")
            index = idx
            
        if not question:
            st.write("Please ask a valid question.")

        # Step 2: Set up the retriever for the vector store
        '''retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": min(100, retriever.docstore.count())}
        )'''

        retriever = vectorstore.as_retriever(search_type="similarity")
        
        doc_count = getattr(vectorstore, 'docstore', {}).get('count', lambda: 0)()
        k = 100
        kvalue = max(k, doc_count)  
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": kvalue}
        )
        def format_docs(docs: Iterable[Document]) -> str:
            """Format documents into a string, ensuring we don't exceed the token limit using the local tokenizer."""
            tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/claude-tokenizer')
            context = ""
            total_tokens = 0
            token_limit = 8192

            for doc in docs:
                # Tokenize the content of the document
                tokens = tokenizer.encode(doc.page_content)
                doc_tokens = len(tokens)

                # If adding this document's tokens would exceed the limit, stop
                if total_tokens + doc_tokens <= token_limit:
                    context += doc.page_content + "\n\n"
                    total_tokens += doc_tokens
                else:
                    break

            print(total_tokens)
            return context

        # Step 4: Define the system prompt with the retrieved context
        template = """You are a specialized assistant who provides information and solutions based exclusively on the database and available support documents. Please translate the documents context from Italian to English to answer in English if the question is asked in English."""

        prompt = ChatPromptTemplate.from_template(template)

        #custom_rag_prompt = PromptTemplate.from_template(template)
        st.session_state['chat_history'].append(("You", question))

        # Get the last 5 relevant bot responses from the chat history
        recent_bot_responses = [message for speaker, message in reversed(st.session_state['chat_history']) if speaker == "Bot"][:5]

        # Step 1: Rephrase the question using the last 5 bot responses
        #question = self.ensure_italian(question)
        if recent_bot_responses:
            rephrase_prompt = f"Rephrase the following question in the context of these recent answers: {', '.join(recent_bot_responses)}. User's question: {question}"
            rephrased_question = self.processor.llm.invoke(rephrase_prompt)
        else:
            rephrased_question = question

        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | self.processor.llm
            | StrOutputParser()
        )

        rag_chain.invoke(question)
        contextualize_q_system_template = """
        Your task is to analyze the user's question:
        {input} 

        and respond appropriately by following these guidelines:
     
        1. Language of the Response:
        - Always must Answer either in Italian or in English in which language the question is asked. 
        - If the question is asked in English then must translate the provided support documents content from Italian to English to answer in English. Otherwise answer in Italian.
        2. Document Analysis:
        - Look for identical or similar components in the documents provided that are present within the question, even if they do not address exactly the same topic.
        - Carefully verify the relevance of the information found.
        3. Formulation of the Response:
        - Do not add your own ideas.
        - Begin your answer directly with the answer to the user's question.
        - Base your response strictly on the information present in the newly retrieved documents, without adding personal interpretations or reasoning.
        - The response must be in the same language the user's question is in, comprehensive, and descriptive.
        - Respond clearly and avoid unnecessary symbols.
        - End the response after providing an answer to the question or stating the impossibility of responding.
        4. In Case of Lack of Information:
        - If you cannot find relevant information or similar components, clearly state that you cannot provide an answer to the question.
        These are the contents you must use to respond to user questions: {context}
        """


        contextualize_q_prompt = ChatPromptTemplate.from_template(contextualize_q_system_template)
        history_aware_retriever = create_history_aware_retriever(
            self.processor.llm, retriever, contextualize_q_prompt
        )

        question_answer_chain = create_stuff_documents_chain(self.processor.llm, contextualize_q_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        @retry_with_exponential_backoff
        def get_response(input_data):
            return rag_chain.invoke(input_data)

        response = get_response({
            "input": rephrased_question,            
            "context": history_aware_retriever,
            "maxTokens": 1024,   
            "temperature": 0.5,
            "chat_history": recent_bot_responses,      
            })
        st.info(f"You: {question}")
        resp = "Bot: "
        with st.empty(): 
             # Create a placeholder for dynamic update
            for chunk in self.stream_response(response['answer']):
                if st.empty():
                    st.info(resp + chunk)                 
                else:
                    st.info(chunk)
       

        
        # Step 11: Option to continue or end the conversation
        st.session_state['chat_history'].append(("Bot", response['answer']))
       

        if 'context' in response and len(response['context']) > 0:
            print(len(response['context']))
            print(response['context'])
            with st.empty():
                src = "Sources of Context: "
                for context_item in response['context']:
                    source = context_item.metadata.get("source", context_item.metadata.get("filename"))
                    page_no = context_item.metadata.get("page_number", None)
                    print(source, page_no)
                    #context = context_item.page_content
                    
                    #st.info(src + '\n' + source + '\n' + "Page Number: " + str(page_no))
                    if len(st.session_state['context_history']) <= index:
                        # Extend the list with empty lists until the index is reached
                         st.session_state['context_history'].extend([[] for _ in range(index - len(st.session_state['context_history']) + 1)])
                    st.session_state['context_history'][index].append(('Context', source + '\n' + "Page Number: " + str(page_no)))
        if not self.is_greeting(question) and not self.is_irrelevant(response['answer']):
            if index < len(st.session_state['context_history']):
                # Extract unique entries based on the last index value (e.g., page number)
                unique_contexts = []
                seen_values = set()

                for context in st.session_state['context_history'][index]:
                    # Get the last value (e.g., page number)
                    last_value = context[1].split()[-1]
                    print(last_value)
                    
                    if last_value not in seen_values:
                        unique_contexts.append(context)
                        seen_values.add(last_value)

                    # Stop if we already have 3 unique entries
                    if len(unique_contexts) == 3:
                        break

            # Display the first 3 unique entries
            print(unique_contexts)
            st.info('First 3 unique ' + src + '\n')
            for unique_context in unique_contexts:
                st.info(unique_context[1])
        
    def ensure_italian(self, text):
        #while detect(text) == 'en':
        translated_text = GoogleTranslator(source='en', target='it').translate(text)
        return translated_text
 



    def process_answer(self, st, query=None, max_tokens=1000):
        st.info("Welcome! You can start chatting with me.")

        # Initialize session state for tracking button state
        if 'button_clicked' not in st.session_state:
            st.session_state.button_clicked = False

        def disable():
            st.session_state.button_clicked = True

        def enable():
            if "button_clicked" in st.session_state and st.session_state.button_clicked == True:
                st.session_state.button_clicked = False
            elif "button_clicked" in st.session_state and st.session_state.button_clicked == False:
                st.session_state.button_clicked = False
               
            

        with st.form("bot_form", clear_on_submit=True):
            if not query:
                # Get user input
                if 'user_input' not in st.session_state:
                    st.session_state['user_input'] = "" 
                user_input = st.text_input("You: ", key="user_input")
                query = user_input
            elif user_input.lower() == 'exit':
                st.info("Ending Conversation")
                return

            # Submit button
            submitted = st.form_submit_button(
                "Submit", 
                on_click=enable, 
                disabled=st.session_state.button_clicked
            )

            if submitted:
                # The button is now disabled; process the query
                with st.spinner("Processing your answer..."):
                    disable()
                    # Simulate processing (replace with your logic)
                    vector_store = self.processor.load_or_initialize_vector_store()
                    if vector_store:
                        pass
                    else:
                        print("Error: Vector Store not found! Creating and loading...")
                        chunks = self.processor.process_pptx_data()
                        vector_store = self.processor.generate_embedding(chunks)
                    self.qa_pipeline(st, vector_store, query)

                # Re-enable the button after processing
         
            st.session_state.button_clicked = False
            enable()


 # Define the retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 5,
    timeout: int = 120,
    errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        start_time = time.time()  # Record start time
        while True:
            try:
                result = func(*args, **kwargs)
                #print(f"Result: {result}")
                if result:
                    return result
            except errors as e:
                print(f"Error: {e}")
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")
                    return None  # Return None if max retries exceeded

                # Check if timeout is exceeded
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout:
                    print(f"Operation timed out after {timeout} seconds. Returning None.")
                    return None  # Return None if timeout exceeded
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)
            except Exception as e:
                raise e
        return result
    return wrapper