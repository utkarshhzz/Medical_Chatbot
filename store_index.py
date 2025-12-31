from dotenv import load_dotenv
import os
from src.helper import load_pdf_files,filter_to_minimal_docs,text_split,download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables first
load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GEMINI_API_KEY=os.environ.get('GEMINI_API_KEY')

# Load and process documents
extracted_data=load_pdf_files(data='data/')
filter_data=filter_to_minimal_docs(extracted_data)
text_chunks=text_split(filter_data)

# Initialize embeddings
embeddings=download_embeddings()

# Initialize Pinecone with correct variable name
pc=Pinecone(api_key=PINECONE_API_KEY)

index_name="medical-bot"

# Check if index exists using list_indexes()
existing_indexes = [idx['name'] for idx in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1")
    )
    
index=pc.Index(index_name)

# Create vector store and upload documents
docsearch=PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print(f"Vector store created successfully with {len(text_chunks)} documents!")
