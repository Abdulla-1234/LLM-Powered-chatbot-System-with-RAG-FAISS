from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Fast Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"], 
    temperature=0.1,
    max_tokens=500  # Limit response length for speed
)

# Lightweight embeddings for speed
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False, 'batch_size': 32}
)

vectordb_file_path = "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt")
    data = loader.load()
    
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)
    print("Knowledge base created successfully!")

def get_quick_answer(question):
    """Fast Q&A without complex chains"""
    try:
        # Load vector database
        vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)
        
        # Get similar documents quickly
        docs = vectordb.similarity_search(question, k=2)  # Only top 2 for speed
        
        if not docs:
            return "I don't know. Please create the knowledge base first.", []
        
        # Combine context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Simple prompt for faster processing
        prompt = f"""Based on the following context, answer the question concisely:

Context: {context}

Question: {question}

Answer (if not found in context, say "I don't know"):"""
        
        # Get response from Gemini
        response = llm.invoke(prompt)
        
        return response.content, docs
        
    except Exception as e:
        return f"Error: {str(e)}", []

def get_qa_chain():
    """Fallback method using the old approach"""
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)
    
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}  # Reduced for speed
    )
    
    prompt_template = """Answer based on context only. Be concise.
    
    Context: {context}
    Question: {question}
    
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return chain

if __name__ == "__main__":
    create_vector_db()
    answer, docs = get_quick_answer("Do you have javascript course?")
    print("Answer:", answer)