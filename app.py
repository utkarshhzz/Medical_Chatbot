from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from src.helper import download_embeddings
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GEMINI_API_KEY=os.environ.get('GEMINI_API_KEY')

embeddings=download_embeddings()

pc=Pinecone(api_key=PINECONE_API_KEY)
index_name="medical-bot"

docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever=docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})

chatModel=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3
)

question=input("Enter your question: ")
docs=retriever.get_relevant_documents(question)
response=chatModel.invoke(question)
print(response.content)
