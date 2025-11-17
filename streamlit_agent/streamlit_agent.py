import streamlit as st
import os
import matplotlib
import matplotlib.pyplot as plt
import re 
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchResults, ArxivQueryRun

# Fix for Matplotlib backend in headless environments like Streamlit
matplotlib.use('Agg') 
CHART_FILE_NAME = "chart.png" 

# --- 1. Tool Definitions ---

# Initialize Tool Wrappers
arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=5,
    doc_content_chars_max=1000
)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_api = WikipediaAPIWrapper()
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)

ddg_tool = DuckDuckGoSearchResults()

# Define the final list of tool objects
TOOLS = [
    wiki_tool,
    ddg_tool,
    arxiv_tool,
    Tool(
        name="python_repl",
        func=PythonREPL().run,
        description=(
            "A Python shell. Use this for code execution, math, and data plotting. "
            "For multi-line code, use newline characters (\\n). **The tool input MUST contain only the Python code and nothing else.**"
            "For plotting, you **MUST** use `matplotlib.pyplot.savefig('chart.png')`. "
            "DO NOT use any other file name like 'msft_7d.png'. " 
            "Then, **mention 'chart.png' in your final response** so the plot is displayed. Do NOT use plt.show()."
        )
    ),
    YahooFinanceNewsTool()
]

# --- 2. Agent Configuration ---

def get_system_message(tools):
    # Dynamically fetch the definitive tool names from the objects (ensures correct names)
    tool_names = {t.name: t.name for t in tools}

    return (
        "You are charlie, a helpful intelligent agent that selects the most appropriate tool for each query and provides the best response."
        "Crucially, you **must only use the tools provided** in your toolset."
        # Using the actual names confirmed in our debugging sessions
        f"Your available tools are: **{tool_names.get('YahooFinanceNewsTool', 'yahoo_finance_news')}**, **{tool_names.get('wikipedia', 'wikipedia')}**, **{tool_names.get('DuckDuckGo Search', 'duck_duck_go_search')}**, **{tool_names.get('arxiv', 'arxiv')}**, and **{tool_names.get('python_repl', 'python_repl')}**."
        "\n\n**Tool Selection Guide:**"
        f"If the query asks for financial news or about share, use the **{tool_names.get('YahooFinanceNewsTool', 'yahoo_finance_news')}** tool."
        f"For general knowledge or web search, use the **{tool_names.get('DuckDuckGo Search', 'duck_duck_go_search')}** tool. Do NOT use 'brave_search'."
        f"For academic/research papers, use the **{tool_names.get('arxiv', 'arxiv')}** tool."
        f"For general encyclopedic knowledge, use the **{tool_names.get('wikipedia', 'wikipedia')}** tool."
        f"For executing code or calculations, use the **{tool_names.get('python_repl', 'python_repl')}** tool."
        "Always decide based on the context of the user's request."
    )

def create_llm(api_key: str, model: str):
    os.environ["GROQ_API_KEY"] = api_key
    # Switch to the most reliable model for tool-use
    return ChatGroq(model=model, temperature=0)

def initialize_agent(llm, tools, system_prompt):
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

# --- 3. Streamlit App Layout ---

st.set_page_config(page_title="Charlie: The Intelligent Groq Agent", layout="wide")
st.title("🤖 Charlie: The Intelligent Groq Agent")
st.caption("Powered by Groq's Llama 3.3 70B and LangChain")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key Input
    groq_api_key = st.text_input("Groq API Key", type="password")
    
    # FIX: Use the reliable 70B model
    model_name= "llama-3.3-70b-versatile" 
    
    st.subheader("Available Tools")
    tool_list = [f"* **{tool.name}**: {tool.description.split('.')[0]}." for tool in TOOLS]
    st.markdown("\n".join(tool_list))

# Main area for user interaction
st.header("Ask Charlie a Question")
query = st.text_area(
    "Enter your query (e.g., 'What is 15 factorial? [python_repl]', 'Latest news on Tesla stock? [yahoo_finance_news]', 'What is the capital of Peru? [wikipedia]')",
    height=150
)
submit_button = st.button("Get Answer", type="primary")

if submit_button:
    # 0. Clear any previous plot file (including the disobedient one, if it exists)
    if os.path.exists(CHART_FILE_NAME):
        os.remove(CHART_FILE_NAME)
    if os.path.exists("msft_7d.png"):
        os.remove("msft_7d.png")


    if not groq_api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
    elif not query:
        st.warning("Please enter a query.")
    else:
        # Define a variable to hold the final file name to check
        plot_file_to_check = CHART_FILE_NAME
        
        with st.spinner(f"Running agent with {model_name}..."):
            try:
                # 1. Initialize LLM
                llm = create_llm(groq_api_key, model_name)
                
                # 2. Get the definitive system prompt
                system_prompt = get_system_message(TOOLS)
                
                # 3. Initialize Agent
                agent_groq = initialize_agent(llm, TOOLS, system_prompt)

                # 4. Invoke Agent
                result_groq = agent_groq.invoke(
                    {"messages": [{"role": "user", "content": query}]}
                )

                # 5. Extract and display the final answer
                final_message_content = result_groq["messages"][-1].content
                
                # --- NEW: TOOL USAGE EXTRACTION ---
                tools_used = set()
                for msg in result_groq.get("messages", []):
                    # LangChain messages that are tool calls have a tool_calls attribute
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for call in msg.tool_calls:
                            # Extract the name of the tool called
                            if hasattr(call, 'name'):
                                tools_used.add(call.name)
                            elif isinstance(call, dict) and 'name' in call:
                                tools_used.add(call['name'])
                
                
                st.subheader("✅ Agent's Final Answer")
                
                # Display the tools used
                if tools_used:
                    tool_list_str = ", ".join([f"`{t}`" for t in sorted(list(tools_used))])
                    st.success(f"🛠️ Tools Used: {tool_list_str}")
                else:
                    st.info("No external tools were used for this query.")
                # --- END NEW TOOL USAGE EXTRACTION ---
                
                
                # --- ROBUST PLOT CHECK LOGIC ---
                # A. Check for the fixed filename first (best case)
                if os.path.exists(CHART_FILE_NAME):
                    plot_file_to_check = CHART_FILE_NAME
                else:
                    # B. FALLBACK: Check the text output for any .png file mentioned 
                    match = re.search(r'([a-zA-Z0-9_\-]+\.png)', final_message_content)
                    if match:
                        potential_file = match.group(1)
                        if os.path.exists(potential_file):
                            plot_file_to_check = potential_file

                # C. Display the final image if found
                if os.path.exists(plot_file_to_check):
                    st.image(plot_file_to_check, caption=f"Plot generated by Python REPL: {plot_file_to_check}")
                
                # Display the text response
                st.info(final_message_content)

            except Exception as e:
                error_message = str(e)
                st.error(f"An error occurred during agent execution.")
                
                # Provide debug hint for common tool error
                if "tool call validation failed" in error_message or "invalid_request_error" in error_message:
                    st.exception(f"Tool Validation Error (Hint: The agent failed to format the JSON tool call argument, likely due to an extra closing brace or improper escaping.):\n\n{error_message}")
                else:
                    st.exception(e)