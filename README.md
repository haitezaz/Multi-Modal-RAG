🔍 Multi-Modal RAG Pipeline with LangChain, Gemini, and FastAPI
This project is a production-ready Retrieval-Augmented Generation (RAG) system that handles multi-modal documents (text, images, tables) using the latest open-source and commercial tools. It extracts, summarizes, stores, retrieves, and responds to queries using both vision and language models.

🚀 Features
🔤 Partition PDFs using unstructured to extract:

Text

Tables

Images (via tesseract OCR)

🤖 Summarize each content type (text, image, table) using Gemini Vision (gemini-1.5-flash)

🧠 Embeddings via HuggingFace (all-MiniLM-L6-v2)

📦 Storage:

Vector DB: ChromaDB

Document Store: MongoDB via LangChain Community MongoDBStore

🔗 MultiVectorRetriever (LangChain): Embed summaries, retrieve raw content

🧠 LLM RAG Chain: Built using LangChain runnables and Google Gemini Vision model

🖼️ Image support via base64 encoding

🛠️ Backend API using FastAPI

🧰 Tech Stack
Component	Tool / Library
Embedding Model	HuggingFace Transformers (MiniLM-L6)
Vision LLM	Google Gemini Vision (1.5 Flash)
PDF Parsing	unstructured, pdf2image, pytesseract
Storage	ChromaDB + MongoDB Atlas
Framework	LangChain + FastAPI
OCR Engine	Tesseract

📁 Folder Structure
graphql
Copy
Edit
.
├── chroma_db/                  # Vector DB storage (persisted)
├── app.py                      # FastAPI app using LangChain chain
├── backend_logic.py            # Chain + retriever setup
├── extracted_data/             # Folder for extracted images
├── utils/                      # Resize, base64, OCR, parsing
├── requirements.txt
└── README.md
📌 How It Works
PDF Parsing:

Uses partition_pdf(strategy="hi_res") to extract structured content.

Images and tables are extracted as .png and passed through Tesseract for OCR when needed.

Summarization:

Text and image summaries are generated using Gemini Vision.

These summaries are indexed as embeddings for retrieval.

Retrieval:

On query, LangChain’s MultiVectorRetriever retrieves relevant summaries → uses doc_id to fetch the full original content (text/table/image) from MongoDB.

LLM Chain:

A custom prompt formats both image and text together.

Gemini responds with rich, context-aware answers.

🧪 Run Locally
Clone the repo

bash
Copy
Edit
git clone https://github.com/yourusername/multi-modal-rag
cd multi-modal-rag
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the API

bash
Copy
Edit
uvicorn app:app --reload
✅ Requirements
Python 3.9+

Tesseract installed and added to PATH

Google Gemini API Key (1.5 Flash)

MongoDB Atlas Cluster URI

Poppler (for pdf2image)

📌 Notes
Images are resized to 1300x600 before inference.

Tables can be treated either as text (via OCR) or passed as base64 images.

You can easily switch to OpenAI or other LLMs with LangChain integration.

📫 Contact
Feel free to connect on www.linkedin.com/in/haitezaz or raise an issue for feedback.

