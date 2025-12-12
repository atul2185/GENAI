from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()
llm=ChatGroq(model="llama-3.1-8b-instant",temperature=0)
system_prompt=SystemMessagePromptTemplate.from_template("You are an invnetory management exepert. Generate Economic Order Quantity (EOQ) based on user inputs.")
user_prompt=HumanMessagePromptTemplate.from_template("Calculate EOQ for this {product}?" \
" Annual Demand: {demand}, Ordering Cost: {ordering_cost}, Holding Cost: {holding_cost}." \
" Provide only the EOQ value and reasoning.")
chat_prompt=ChatPromptTemplate.from_messages([system_prompt,user_prompt])
chain=chat_prompt | llm
response=chain.invoke({"product":"laptops","demand":1200,"ordering_cost":50,"holding_cost":10})
print(response.content)

