import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.set_page_config(page_title="ChartBot Q&A")
st.title("ChartBot Q&A 🌱")

# Create the knowledge base button
if st.button("📚 Create Knowledgebase"):
    with st.spinner("🔄 Creating knowledge base..."):
        try:
            create_vector_db()
            st.success("✅ Knowledge base created successfully!")
        except Exception as e:
            st.error(f"❌ Failed to create knowledge base: {e}")

# Question input box
question = st.text_input("Enter your question:")

# Handle question submission
if question:
    with st.spinner("🔍 Searching for answer..."):
        try:
            chain = get_qa_chain()
            response = chain.invoke({"query": question})

            # Display answer
            st.subheader("✅ Answer")
            st.write(response["result"])

            # Optional: show source documents
            if st.checkbox("📄 Show source documents"):
                st.markdown("---")
                for i, doc in enumerate(response["source_documents"]):
                    st.markdown(f"**🔹 Source {i + 1}:**")
                    st.code(doc.page_content)
                    st.markdown("---")
        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")
            st.info("📌 Make sure you've created the knowledge base before asking questions.")
