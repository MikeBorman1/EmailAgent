
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
import os
import pandas as pd
from dotenv import load_dotenv
import pinecone
from langchain.vectorstores import Pinecone
from langchain.tools import Tool
from fastapi import FastAPI
from pydantic import BaseModel
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_enviroment = os.getenv("PINECONE_ENVIRONMENT")
# Initialize Pinecone client
pinecone.init(api_key=pinecone_api_key, environment=pinecone_enviroment) 



embeddings = OpenAIEmbeddings()
# Create Pinecone vectorstore instance 
vectorstore = Pinecone.from_existing_index(index_name="new", embedding=embeddings, text_key= "text",namespace="salesdocs2")

retriever = vectorstore.as_retriever()



llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
# Create QA chain with retriever



# Create QA chain
qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

def run_qa(question):
    return qa_chain.run({"query": question})

def search_orders(email):
    """
    Search email against orders dataframe to find matching rows
    
    Args:
        email (str): Email contents to search against orders
        
    Returns:
        matching_rows (DataFrame): Subset of orders dataframe containing matching rows
    """
    print(email)
    orders_df = pd.DataFrame({
        'email': ['mikeborman.ada@gmail.com', 'jane@example.com', 'bob@example.net'],
        'order_id': ['1234', '5678', '9012'],
        'customer': ['Mike Doe', 'Jane Doe', 'Bob Smith'],
        'product': ['T-shirt', 'Pants', 'Shoes'],
        'shipped': [True, False, False],
        'quantity': [1, 2, 3]
    })
    
    # Search email text against order_id column
    # Use regex to match full email address instead of contains
    matching_rows = orders_df[orders_df['email'].str.match(email)]

    return str(matching_rows)

orders_tool = Tool.from_function(
  name="Orders_DB",
  description="Usefull when needing to look up somebodys orders with their email",
  func=search_orders
)

qa_tool = Tool.from_function(
  name="Buisiness_docs",
  description="all of the documents and general info about the buisiness - usefull for answering general questions ",
  func=run_qa
)

tools = [qa_tool,orders_tool]

system_message = SystemMessage(
    content = """
You are an exceptional customer service agent, responsible for handling and responding to customer emails with a focus on providing accurate and timely information. You have access to the company's internal systems, allowing you to review order details, track shipments, and access a wealth of business information to address customer queries.

Your Job is to respond the the current email given to you.

Please ensure you follow these guidelines when responding to customer emails:

1/ Thoroughly review the customer's email and any attached information to understand their query, complaint, or request. Do not ignore any details provided by the customer.

2/ Access the company's internal systems to verify and gather the necessary information relevant to the customer's query. For example, if the customer is inquiring about an order, check the order details, payment status, and shipment tracking.

3/ Structure youre resposne in HTML with HTML tags for the line breaks

4/ After obtaining all the necessary information from the functions available to you, formulate a clear and concise response to the customer's email. Ensure that your response addresses the customer's specific query, offers a solution, or provides relevant information.

5/ Provide a comprehensive explanation and avoid vague statements. 

6/ Include any relevant documentation, links, or reference numbers that may be helpful to the customer, such as order confirmations, shipping tracking numbers, or company policies.

7/ If the customer's issue requires escalation to another department or involves a more complex resolution, inform the customer of the steps you have taken, the expected timeline for resolution, and who will be in contact with them.

8/ Proofread your response before sending it to the customer. Ensure that it is free of typos and grammatical errors and that it reflects a professional and courteous tone.

9/ Always consider the customer's perspective and strive to exceed their expectations. Make sure your response demonstrates empathy and understanding.

Remember you are responding to the single email, SO ONLY OUTPUT A SINGLE EMAIL


    """)
agent_kwargs = {
    "system_message": system_message,
}

memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=2000)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)



#Set this as an API endpoint via FastAPI
app = FastAPI()


class Query(BaseModel):
    query: str


@app.post("/")
def researchAgent( query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content['output']
    return actual_content