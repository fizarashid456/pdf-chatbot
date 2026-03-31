# 📄 PDF Chatbot — Powered by Google Gemini AI

An intelligent PDF chatbot built with **Streamlit**, **LangChain**, and **Google Gemini AI**. Upload any PDF and have a natural conversation about its contents — completely free using Google's Gemini 1.5 Flash model.

---

## ✨ Features

- 📂 **Upload any PDF** — research papers, books, manuals, reports
- 🤖 **Gemini 1.5 Flash** — fast, intelligent responses from Google's free AI
- 🧠 **Semantic search** — HuggingFace embeddings for accurate retrieval
- 💬 **Chat interface** — full conversation history with context
- 📚 **Source references** — see exactly which parts of the document were used
- 🎛️ **Configurable settings** — temperature, chunk size, number of sources
- 🌑 **Dark-themed UI** — clean, modern interface

---

## 🔧 Prerequisites

- Python 3.8 or higher
- A free Google Gemini API key

---

## 🔑 Getting a Free Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key (starts with `AIza...`)

The free tier includes generous usage limits — no credit card required.

---

## 🚀 Installation

### 1. Clone or download this project

```bash
git clone https://github.com/yourname/pdf-chatbot.git
cd pdf-chatbot
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ First run downloads ~90MB of HuggingFace embedding models. This is a one-time download.

### 4. Set up your API key

```bash
# Copy the example file
cp .env.example .env

# Edit .env and replace with your actual key
GOOGLE_API_KEY=AIzaSy...your-key-here
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 📖 How to Use

1. **Enter your API key** in the left sidebar (or set it in `.env`)
2. **Click "Validate Key"** to confirm it works
3. **Upload a PDF** using the file uploader
4. **Click "Process PDF"** — wait for chunking and indexing
5. **Ask questions** in the chat box at the bottom
6. Toggle **"Show source references"** to see which document sections were used

---

## ⚙️ Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| Temperature | Controls response creativity (0 = precise, 1 = creative) | 0.3 |
| Sources to retrieve | Number of document chunks used as context | 4 |
| Show source references | Display the source text chunks used | Off |
| Chunk size | Size of each text chunk in characters | 800 |
| Chunk overlap | Overlap between adjacent chunks | 100 |

---

## 🗂️ Project Structure

```
pdf-chatbot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
├── .env                # Your actual API key (create this, don't commit!)
└── README.md           # This file
```

---

## 🛠️ Troubleshooting

### "Invalid API key" error
- Make sure you copied the full key including the `AIza` prefix
- Check that the key is enabled in Google AI Studio
- Ensure you haven't exceeded your free quota

### PDF processing fails
- Ensure the PDF is not password-protected
- Try a smaller PDF first (< 50 pages) to test
- Some scanned PDFs without OCR text may not work

### Slow first run
- The HuggingFace model (`all-MiniLM-L6-v2`) downloads ~90MB on first use
- Subsequent runs use the cached model and are much faster

### "Module not found" errors
- Make sure your virtual environment is activated
- Re-run `pip install -r requirements.txt`
- For `chromadb` issues on Windows, you may need: `pip install chromadb --no-binary chromadb`

### Out of memory errors
- Reduce the chunk size in the sidebar
- Process a smaller PDF
- Restart the Streamlit app to clear memory

---

## 🧰 Tech Stack

| Component | Library |
|-----------|---------|
| UI Framework | Streamlit |
| AI Orchestration | LangChain 0.3+ |
| Language Model | Google Gemini 1.5 Flash |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| PDF Loading | PyPDF |

---

## 📜 License

MIT License — free to use and modify.

---

*Built with ❤️ using Streamlit · LangChain · Google Gemini AI*
