# LLM-Powered-chatbot-System-with-RAG-FAISS
# ChartBot Q&A System 🌱

An intelligent Question & Answer system built with **LangChain**, **Google Gemini**, and **Streamlit** that provides instant answers to course-related queries using advanced **Retrieval-Augmented Generation (RAG)** technology.

<p align="center">
  <img src="https://github.com/Abdulla-1234/LLM-Powered-chatbot-System-with-RAG-FAISS/blob/main/ChartBot_QA_Sample2.jpeg" alt="System Architecture" width="50%" style="margin-right: 10px;" />
  <img src="https://github.com/Abdulla-1234/LLM-Powered-chatbot-System-with-RAG-FAISS/blob/main/ChartBot_QA_Sample3.jpeg" alt="UI Screenshot" width="47%" />
</p>


## Features

- **Lightning Fast Responses**: Get answers in 1-2 seconds using Google Gemini 1.5 Flash
- **Smart Document Search**: Uses FAISS vector database for efficient similarity search
- **Interactive Web Interface**: Clean, user-friendly Streamlit interface
- **Source Transparency**: View source documents for answer verification
- **Real-time Response Tracking**: See exact response times
- **Knowledge Base Management**: Easy creation and updates of the knowledge base

## Technologies Used

- **LangChain**: Framework for building LLM applications
- **Google Gemini 1.5 Flash**: Fast and efficient language model
- **FAISS**: Facebook AI Similarity Search for vector operations
- **Streamlit**: Web application framework
- **HuggingFace Transformers**: For text embeddings
- **Sentence Transformers**: Semantic text similarity
- **Python**: Core programming language

## Architecture 
<img src="https://github.com/Abdulla-1234/LLM-Powered-chatbot-System-with-RAG-FAISS/blob/main/architecture.jpeg" alt="System Architecture Diagram" width="400"/>

## Project Structure

```
LLM-Powered-chatbot-System-with-RAG-FAISS/
├── langchain_helper.py      # Core RAG implementation
├── main.py                  # Streamlit web application
├── codebasics_faqs.csv     # Knowledge base (FAQ data)
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys)
├── faiss_index/           # Vector database storage
│   ├── index.faiss
│   └── index.pkl
└── README.md              # Project documentation
```

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Abdulla-1234/LLM-Powered-chatbot-System-with-RAG-FAISS.git
cd LLM-Powered-chatbot-System-with-RAG-FAISS
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
# Create .env file
echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
```

**Get Google API Key:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Replace `your_google_api_key_here` with your actual key

### 5. Run the Application
```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

### Creating Knowledge Base
1. Click "🔄 Create/Update Knowledge Base" in the sidebar
2. Wait for the process to complete
3. You'll see "Knowledge base ready!" message

### Asking Questions
1. Type your question in the input field
2. Press Enter or wait for auto-processing
3. Get instant answers with response time display
4. Optionally view source documents for transparency

### Example Questions
- "Can I add this course to my resume? If Yes, how?"
- "Will this bootcamp guarantee me a job? "
- "How can I contact support?"
- "Do you provide certificates?"
- "mention all course"

## Configuration

### Customizing the Knowledge Base
Replace `codebasics_faqs.csv` with your own data:
- **prompt**: Questions/topics column
- **response**: Answers/content column

### Model Settings
Modify in `langchain_helper.py`:
```python
# Change model for different speed/accuracy trade-offs
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # or "gemini-1.5-pro" for higher accuracy
    temperature=0.1,           # Lower = more consistent answers
    max_tokens=500            # Limit response length
)
```

### Performance Tuning
```python
# Adjust retrieval parameters
docs = vectordb.similarity_search(question, k=2)  # Number of documents to retrieve
```

## Deployment

### Local Development
```bash
streamlit run main.py
```

### Streamlit Cloud
1. Push code to GitHub
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add `GOOGLE_API_KEY` to secrets
4. Deploy automatically

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "main.py"]
```

## Performance Metrics

- **Response Time**: 1-2 seconds average
- **Accuracy**: High relevance using semantic search
- **Scalability**: Handles 1000+ FAQ entries efficiently
- **Memory Usage**: ~200MB with loaded models

## Troubleshooting

### Common Issues

**Error: "models/gemini-pro is not found"**
- Solution: Update to `gemini-1.5-flash` in langchain_helper.py

**Slow responses**
- Check internet connection
- Reduce `k` value in similarity search
- Use lighter embedding model

### Performance Tips
- Use SSD storage for vector database
- Increase batch_size for bulk operations
- Consider GPU acceleration for large datasets

## Contact

- **Developer**: D Mohammad Abdulla
- **Email**: mohammadabdulla20march@gmail.com
- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/mohammad-abdulla-doodakula-8a3307258/)
- **GitHub**: [Your GitHub Profile](https://github.com/Abdulla-1234)

---

⭐ **Star this repository if you found it helpful!**
