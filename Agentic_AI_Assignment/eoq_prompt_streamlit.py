import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import os

# --- Configuration and Initialization ---

# Define the LangChain Prompt Template
system_prompt_template = (
    r"You are an inventory management expert. Generate the Economic Order Quantity (EOQ) based on user inputs. "
    r"Provide the final EOQ value and a brief explanation of the calculation/reasoning."
)

human_prompt_template = (
    r"Calculate EOQ for this {product}?\n"
    r"Annual Demand: {demand}, Ordering Cost: {ordering_cost}, Holding Cost: {holding_cost}.\n"
    r"Provide only the EOQ value and the reasoning for the calculation."
)

# --- Streamlit UI ---

st.set_page_config(page_title="EOQ Calculator", layout="centered")

st.title("🛒 Economic Order Quantity (EOQ) Calculator")
st.markdown("Determine the **optimal order quantity** that minimizes total inventory costs.")


# --- API Key Input in Sidebar ---
with st.sidebar:
    st.header("🔑 Groq API Key")
    groq_api_key = st.text_input(
        "Enter your Groq API Key", 
        type="password", 
        help="You can get your key from the Groq Developer Console."
    )
    st.caption("The key is not stored and is only used for this session.")
    st.markdown("---")
    st.header("App Info")
    st.markdown(
        """
        - **Model:** Llama-3.1-8b-instant
        - **Purpose:** Calculates the Economic Order Quantity (EOQ).
        """
    )
    
# --- Main App Inputs ---
with st.form("eoq_form"):
    st.header("Input Parameters")
    
    # Text input for product
    product = st.text_input(
        "Product Name", 
        value="Laptops", 
        help="The specific item for which you are calculating EOQ."
    )
    
    # Number input for demand (D)
    demand = st.number_input(
        "Annual Demand (D)", 
        min_value=1, 
        value=1200, 
        step=100,
        help="The total number of units demanded over a year."
    )
    
    # Number input for ordering cost (S)
    ordering_cost = st.number_input(
        "Ordering Cost (S)", 
        min_value=0.01, 
        value=50.0, 
        step=10.0,
        help="The fixed cost incurred per order (e.g., shipping, handling)."
    )
    
    # Number input for holding cost (H)
    holding_cost = st.number_input(
        "Holding Cost (H)", 
        min_value=0.01, 
        value=10.0, 
        step=1.0,
        help="The cost of holding one unit of inventory for one year."
    )
    
    # Submit button
    submitted = st.form_submit_button("Calculate EOQ")

# --- Logic and Output ---

if submitted:
    # 1. Check for API Key
    if not groq_api_key:
        st.warning("⚠️ Please enter your Groq API Key in the sidebar to run the calculation.")
        st.stop()
        
    # 2. Initialize LLM and Chain *after* confirming the key is available
    try:
        # Initialize LLM with the user-provided key
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, groq_api_key=groq_api_key)

        # Create the LangChain objects
        system_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)
        user_prompt = HumanMessagePromptTemplate.from_template(human_prompt_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        chain = chat_prompt | llm
        
    except Exception as e:
        st.error(f"Error initializing Groq: {e}. Check the key format.")
        st.stop()

    # 3. Prepare and Invoke the Chain
    inputs = {
        "product": product,
        "demand": demand,
        "ordering_cost": ordering_cost,
        "holding_cost": holding_cost
    }
    
    with st.spinner(f"Running EOQ calculation for **{product}** with Groq..."):
        try:
            # Invoke the LangChain/Groq chain
            response = chain.invoke(inputs)
            
            st.success("✅ Calculation Complete")
            st.subheader(f"Optimal Order Quantity for {product}")
            
            # Display the LLM's full response
            st.info(response.content)

        except Exception as e:
            st.error(f"An error occurred during calculation. Please check your API key or inputs. Error: {e}")

# Footer/Formula Reference
st.markdown("---")
st.caption("The Economic Order Quantity (EOQ) formula is:")
st.latex(r'EOQ = \sqrt{\frac{2DS}{H}}') 
st.caption("Where $D$ is Annual Demand, $S$ is Ordering Cost, and $H$ is Holding Cost.")