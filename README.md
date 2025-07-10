📺 YouTube Chatbot Plugin Backend

A backend service for answering questions about YouTube videos using transcripts, LLM workflows (LangChain & LangGraph), vector search (FAISS), and arXiv research search.

✨ Features
✅ Fetch YouTube video transcripts automatically
✅ Summarize entire video content
✅ Answer user questions about the video context
✅ Semantic search over transcript (FAISS)
✅ Academic research paper search from arXiv
✅ Asynchronous FastAPI server

🗂️ Project Structure
graphql
Copy
Edit
youtube-chatbot-plugin/
│
├── main.py                   # FastAPI entrypoint
├── api/
│   ├── youtube_qa_api.py     # YouTube transcript QA endpoints
│   ├── research_tool.py      # arXiv paper search
│   └── ...
├── services/
│   ├── langchain_pipeline.py # LangChain / LangGraph orchestration
│   └── faiss_loader.py       # FAISS vector store integration
├── utils/
│   └── logger.py
├── requirements.txt
└── .env.example
⚙️ Installation
1️⃣ Clone the repository

bash
Copy
Edit
git clone https://github.com/niteesh577/youtube-chatbot-plugin.git
cd youtube-chatbot-plugin
2️⃣ Create and activate virtual environment

bash
Copy
Edit
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
3️⃣ Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Configure environment variables

bash
Copy
Edit
cp .env.example .env
Edit .env with your API keys.

🏃‍♂️ Running the Server
Start the FastAPI server with:

bash
Copy
Edit
uvicorn main:app --reload
API will be available at:

cpp
Copy
Edit
http://127.0.0.1:8000
📌 Example Endpoints
✅ POST /chat
→ Ask a question about a YouTube video, with video ID & timestamp.

✅ POST /summarize
→ Get a full summary of a video transcript.

✅ GET /search-research
→ Search academic papers via arXiv.

💡 Environment Variables
Example .env:

ini
Copy
Edit
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
LANGSMITH_API_KEY=your_langsmith_key
REDIS_URL=redis://localhost:6379
📚 Requirements
All dependencies are in requirements.txt.

Key packages:

fastapi

uvicorn

langchain

langgraph

faiss-cpu

sentence-transformers

youtube-transcript-api

arxiv

python-dotenv

🛠️ Development Tips
✅ Auto-reload server

bash
Copy
Edit
uvicorn main:app --reload
✅ Code formatting

bash
Copy
Edit
black .
✅ Type checking

bash
Copy
Edit
mypy .
🧑‍💻 Contributing
Pull requests are welcome!
Please open an issue first to discuss what you would like to change.

📜 License
This project is licensed under the MIT License.
See LICENSE.

🙏 Acknowledgements
FastAPI

LangChain

LangGraph

SentenceTransformers

FAISS

arXiv API

🌟 Star the Repository!
If you find this project helpful, please ⭐️ the repo to support development!

