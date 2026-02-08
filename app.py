from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

load_dotenv() 

app = Flask(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

embedding = download_huggingface_embeddings()

index_name = "medichat"
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embedding,
    index_name=index_name,
)

retriever = docsearch.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 3}
)

chatModel = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="openai/gpt-4o-mini",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(
    llm=chatModel,
    prompt=prompt
)

rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=question_answer_chain
)



@app.route('/')
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form['msg']
    input = msg
    print(input)
    
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])







if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)