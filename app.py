from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from src.helper import download_embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from flask import Flask,render_template,request,jsonify

load_dotenv()

app=Flask(__name__)

#Initialising components

PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
GEMINI_API_KEY=os.getenv('GEMINI_API_KEY')

#initialising pinecone
pc=Pinecone(api_key=PINECONE_API_KEY)
index_name="medical-bot"

#initialising embeddings
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#initialising vector store
docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


#initialising retreiver
retriever=docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})

chatModel=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3
)
@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/ask',methods=['POST'])
def ask():
    try:
        data=request.json
        question=data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}),400

        # Get relevant documents from Pinecone
        docs=retriever.invoke(question)
        
        # Create context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt with context for better answers
        prompt = f"""Based on the following medical information, answer the question accurately and professionally.

Context:
{context}

Question: {question}

Answer (provide clear, concise medical information):"""
        
        # Get response from Gemini with context
        response=chatModel.invoke(prompt)
        answer=response.content
        
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__=='__main__':
    app.run(debug=True,port=5000)