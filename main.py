import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title("ChartBot Q&A 🌱")

btn = st.button("Create Knowledgebase")
if btn:
    with st.spinner("Creating knowledge base..."):
        create_vector_db()
        st.success("Knowledge base created successfully!")

question = st.text_input("Question: ")

if question:
    with st.spinner("Searching for answer..."):
        try:
            chain = get_qa_chain()
            # Use invoke instead of deprecated __call__
            response = chain.invoke({"query": question})
            
            st.header("Answer")
            st.write(response["result"])
            
            # Optional: Show source documents for transparency
            if st.checkbox("Show source documents"):
                st.subheader("Source Documents")
                for i, doc in enumerate(response["source_documents"]):
                    st.write(f"**Source {i+1}:**")
                    st.write(doc.page_content)
                    st.write("---")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please make sure you have created the knowledge base first.")