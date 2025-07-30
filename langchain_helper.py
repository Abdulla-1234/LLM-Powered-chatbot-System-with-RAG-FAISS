from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# === 1. LLM and Embedding Config ===
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.1,
    max_tokens=500
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False, 'batch_size': 32}
)

vectordb_file_path = "faiss_index"


# === 2. Create Vector Database ===
def create_vector_db():
    csv_path = "codebasics_faqs.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return

    try:
        # Optional debug: Check columns
        df = pd.read_csv(csv_path)
        print(f"📄 CSV columns: {df.columns.tolist()}")
        
        if "prompt" not in df.columns:
            raise ValueError("❌ Column 'prompt' not found in the CSV file.")
        
        loader = CSVLoader(file_path=csv_path, source_column="prompt")
        data = loader.load()
        
    except Exception as e:
        print(f"⚠️ CSV load failed: {e}")
        print("🔁 Using fallback mock data instead.")
        data = [
            Document(page_content="Yes, we offer a JavaScript course.", metadata={}),
            Document(page_content="You can find our courses at codebasics.io.", metadata={})
        ]

    # Save the vector database
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)
    print("✅ Knowledge base created successfully!")


# === 3. Lightweight QA ===
def get_quick_answer(question):
    try:
        vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)
        docs = vectordb.similarity_search(question, k=2)

        if not docs:
            return "I don't know. Please create the knowledge base first.", []

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""Based on the following context, answer the question concisely:

Context: {context}

Question: {question}

Answer (if not found in context, say "I don't know"):"""

        response = llm.invoke(prompt)
        return response.content, docs

    except Exception as e:
        return f"Error: {str(e)}", []


# === 4. Traditional QA Chain (Optional) ===
def get_qa_chain():
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA

    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    prompt_template = """Answer based on context only. Be concise.

Context: {context}
Question: {question}

Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain


# === 5. Main Run Block ===
if __name__ == "__main__":
    create_vector_db()
    question = "Do you have javascript course?"
    answer, docs = get_quick_answer(question)
    print("Answer:", answer)
