import streamlit as st
import pandas as pd
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # A robust, local embedding model
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---

# Define the Embedding Model (lightweight and works locally)
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 
VECTOR_STORE_PATH = "./chroma_db_eoq"

# --- Prompt Templates ---

# The System Prompt is now RAG-aware, asking the LLM to use the provided context (the retrieved data).
system_prompt_template = (
    "You are an expert Inventory Management system. Your task is to calculate the Economic Order Quantity (EOQ) "
    "based *only* on the context provided below. Do not use external knowledge for calculation values.\n"
    "1. **Extract** the Demand, Ordering Cost, and Holding Cost from the 'Context'.\n"
    "2. **Provide** only the final EOQ value and a brief explanation of the calculation/reasoning.\n"
    "--- CONTEXT ---\n"
    "{context}"
)

# The Human Prompt is now just the product name (the query).
human_prompt_template = "Calculate EOQ for the product: {product}."

system_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)
user_prompt = HumanMessagePromptTemplate.from_template(human_prompt_template)
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])


# --- RAG Helper Functions ---

# Function to create text chunks from a DataFrame row
def create_document_chunks(df: pd.DataFrame) -> list:
    """Converts DataFrame rows into descriptive text chunks."""
    documents = []
    # Ensure column names are standardized for robust retrieval
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    
    for index, row in df.iterrows():
        # Adjust these column names to match your actual CSV/Excel file's headers
        try:
            product = row['product_name']
            demand = row['demand']
            ordering_cost = row['ordering_cost']
            holding_cost = row['holding_cost']
            
            # Create a rich text description of the row
            content = (
                f"Product Name: {product}, Annual Demand (D): {demand}, "
                f"Ordering Cost (S): {ordering_cost}, Holding Cost (H): {holding_cost}. "
                "These values are used to calculate the Economic Order Quantity."
            )
            documents.append(content)
        except KeyError as e:
            st.error(f"Missing expected column in the file: {e}. Check that your headers are 'product name', 'demand', 'ordering cost', and 'holding cost'.")
            return []
    return documents

@st.cache_resource
def get_vector_store(text_chunks):
    """Initializes the embedding model and creates/loads the Chroma vector store."""
    if not text_chunks:
        return None
        
    try:
        # 1. Initialize Embeddings
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # 2. Create the Vector Store
        # We don't store documents explicitly here; the chunks are just strings.
        # Chroma expects a list of documents (strings)
        vector_store = Chroma.from_texts(
            texts=text_chunks, 
            embedding=embeddings, 
            persist_directory=VECTOR_STORE_PATH # Persist to disk for faster re-runs
        )
        vector_store.persist()
        st.success("✅ Product data loaded and RAG database built!")
        return vector_store
    except Exception as e:
        st.error(f"Error building RAG database: {e}")
        return None

# --- Streamlit UI and Logic ---

st.set_page_config(page_title="RAG EOQ Calculator", layout="centered")

st.title("RAG-Powered EOQ Calculator")
st.markdown("Upload your product data file to enable semantic EOQ calculation.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("1. API Key")
    groq_api_key = st.text_input(
        "Enter your Groq API Key", 
        type="password", 
        help="Required to run the LLM calculation."
    )
    st.markdown("---")
    
    st.header("2. Product Data (.csv or .xlsx)")
    uploaded_file = st.file_uploader(
        "Upload file (CSV or Excel)", 
        type=["csv", "xlsx"], 
        help="Must contain columns: 'Product Name', 'Demand', 'Ordering Cost', 'Holding Cost'"
    )

# --- Data Processing ---
vector_store = None
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else: # assuming .xlsx
            df = pd.read_excel(uploaded_file)
            
        st.caption("Data Preview:")
        st.dataframe(df.head(2))
        
        # Create chunks and build the vector store
        chunks = create_document_chunks(df)
        if chunks:
            vector_store = get_vector_store(chunks)

    except Exception as e:
        st.error(f"Error reading file: {e}")

# --- Main App Inputs ---
with st.form("eoq_form"):
    st.header("3. Run Calculation")
    
    # Text input for product query
    product_query = st.text_input(
        "Enter Product Query", 
        value="Laptops", 
        help="Enter the exact product name, or a descriptive query (e.g., 'Best selling PC')."
    )
    
    submitted = st.form_submit_button("Calculate EOQ via RAG")

# --- RAG Logic and Output ---

if submitted:
    if not groq_api_key:
        st.warning("⚠️ Please enter your Groq API Key in the sidebar.")
        st.stop()
    if not vector_store:
        st.warning("⚠️ Please upload a valid product data file first.")
        st.stop()
        
    # 1. Initialize LLM and Retriever
    with st.spinner("Initializing LLM and RAG components..."):
        try:
            llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, groq_api_key=groq_api_key)
            retriever = vector_store.as_retriever(search_kwargs={"k": 1}) # Retrieve the single best match

        except Exception as e:
            st.error(f"Error initializing Groq LLM: {e}. Check your key.")
            st.stop()
            
    # 2. Build the RAG Chain
    rag_chain = (
        {"context": retriever, "product": RunnablePassthrough()} 
        | chat_prompt 
        | llm 
        | StrOutputParser()
    )

    # 3. Invoke the RAG Chain
    with st.spinner(f"Searching for data for '{product_query}' and calculating EOQ..."):
        try:
            # The input to the chain is the user's query
            response = rag_chain.invoke(product_query)
            
            st.success("✅ Calculation Complete")
            st.subheader(f"EOQ for: {product_query}")
            st.info(response)

        except Exception as e:
            st.error(f"An error occurred during RAG calculation: {e}")

# Footer/Formula Reference
st.markdown("---")
st.caption("The Economic Order Quantity (EOQ) formula is:")
st.latex(r'EOQ = \sqrt{\frac{2DS}{H}}') 
st.caption("This application uses a RAG pipeline (HuggingFace Embeddings + ChromaDB) to retrieve product data from the uploaded file before asking the Groq LLM to perform the calculation.")