import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
# Import itemgetter for more stable LCEL input mapping
from operator import itemgetter

# --- Utility Functions ---

def process_pdf(file_path):
    """Loads PDF, creates embeddings, and initializes a Chroma vector store."""
    try:
        # Attempt PyPDFLoader first, fall back to Unstructured if it fails
        try:
            loader=PyPDFLoader(file_path)
            documents=loader.load()
        except:
            loader=UnstructuredPDFLoader(file_path,strategy="fast")
            documents=loader.load()

        # Define embedding model
        embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create vector store
        return Chroma.from_documents(documents,embeddings)
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

def format_docs(docs):
    """Formats retrieved documents into a single string context."""
    return "\n\n".join([doc.page_content for doc in docs])

def format_chat_history(chat_history):
    """Formats the Streamlit session chat history into a string for the prompt."""
    if not chat_history:
        return "No Previous Conversation"
    formatted=""
    # chat_history is a list of (query, answer) tuples
    for q,a in chat_history:
        # Note: We must ensure 'a' is a string before appending
        answer_text = str(a)
        formatted+=f"human: {q}\nAssistant: {answer_text}\n"
    return formatted if formatted else "No Previous Conversation"

# --- Main Chain Creation Function ---

def create_conversational_chain (vector_store,groq_api_key):
    """Creates the LangChain RAG chain with conversational memory."""
    llm=ChatGroq(model="llama-3.1-8b-instant",temperature=0,api_key=groq_api_key)

    # Retriever setup
    retriever=vector_store.as_retriever(search_kwargs={"k":3})

    # Create prompt template
    prompt=ChatPromptTemplate.from_messages([
        ("system","""You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
         to answer the question. Consider the chat history for context but focus on answering the current question.If you 
         don't know the answer, just say that you don't know. Keep your answers concise and to the point.
         
         context from PDF : {context}
         previous conversation history : {chat_history}
         """),
         ("human", "{question}")
    ])
    
    # 🚨 Using itemgetter for maximum stability in input mapping 🚨
    chain = (
        # Ensure the input dictionary is explicitly handled
        RunnablePassthrough() 
        | RunnablePassthrough.assign(
            # Generate the context from the retriever
            context=itemgetter("question") | retriever | format_docs, 
            # Pass through the original input keys using itemgetter
            question=itemgetter("question"),
            chat_history=itemgetter("chat_history"),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --- Streamlit Main Function ---

def main():
    st.set_page_config(page_title="Chat with PDF")

    # Initialize Session State variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store=None
    if "conversational_chain" not in st.session_state:
        st.session_state.conversational_chain=None
        
    st.title("Chat with PDF :blue[🚀]")
    
    # Sidebar for setup
    with st.sidebar:
        st.header("Upload your PDF")
        uploaded_file=st.file_uploader("Choose a PDF file",type="pdf")
        # Ensure API key input is available
        groq_api_key=st.text_input("Enter your Groq API Key",type="password")
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history=[]
            st.rerun()

        # PDF and Chain Initialization Logic
        if uploaded_file and not st.session_state.vector_store:
            with st.spinner("Processing PDF..."):
                # Save uploaded file temporarily
                temp_file_path = "temp.pdf"
                with open(temp_file_path,"wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.session_state.vector_store=process_pdf(temp_file_path)
                
                # Cleanup temporary file
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except Exception as e:
                        print(f"Failed to remove temp file: {e}")


            if st.session_state.vector_store and groq_api_key:
                st.session_state.conversational_chain=create_conversational_chain(st.session_state.vector_store,groq_api_key)
                st.success("PDF processed successfully! Ready to chat.")
            elif st.session_state.vector_store:
                st.warning("PDF processed. Please enter your Groq API Key to start the chatbot.")
            else:
                st.error("Failed to process PDF.")

    # Main Chat Interface
    if st.session_state.vector_store and groq_api_key:
        # Re-create chain if key was entered later
        if not st.session_state.conversational_chain:
            st.session_state.conversational_chain=create_conversational_chain(st.session_state.vector_store,groq_api_key)
            
        # Display existing chat history
        for query,answer in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                st.write(answer)
                
        # Handle new user queries
        query=st.chat_input("Ask a question about your PDF document")
        if query:
            with st.chat_message("user"):
                st.write(query)
            
            try:
                # Format history for the prompt
                chat_histoty_formatted=format_chat_history(st.session_state.chat_history)
                
                with st.chat_message("assistant"):
                    with st.spinner("Generating response..."):
                        # Invoke the chain with the two required inputs
                        result=st.session_state.conversational_chain.invoke({
                            "question":query,
                            "chat_history":chat_histoty_formatted
                        })
                        
                        # Since StrOutputParser is used, result should be a string
                        answer = str(result)
                        st.write(answer)

                st.session_state.chat_history.append((query, answer))
            except Exception as e:
                # Catch the Groq API error or any other chain failure
                st.error(f"Error generating response: {str(e)}")

    else:
        st.info("Please upload a PDF and enter your Groq API Key to start chatting.")

# Final cleanup of the temp file (though handled in sidebar, better to be safe)
    if os.path.exists("temp.pdf"):
        try:
            os.remove("temp.pdf")
        except:
            pass

if __name__=="__main__":
    main()
