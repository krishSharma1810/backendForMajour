from flask import Flask, request, jsonify, g
from flask_cors import CORS
import os
import faiss
import pickle
import pdfplumber
from werkzeug.utils import secure_filename
from langchain_community.vectorstores import FAISS  # ✅ Using FAISS instead of Astra DB
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ Updated import
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from googleapiclient.discovery import build
from duckduckgo_search import DDGS
from dotenv import load_dotenv

# ✅ Load Environment Variables
load_dotenv()

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)

# ✅ Directory to store uploaded PDFs
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Load API Keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

# ✅ Use FAISS Instead of Cassandra (Fixes High Memory Usage)
def initialize_vector_store(text_chunks):
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")  # ✅ Lighter Model
    vector_store = FAISS.from_texts(text_chunks, embedding)
    
    # ✅ Save to Disk (Avoid Memory Overflow)
    faiss.write_index(vector_store.index, "vector_store.index")
    with open("vector_store.pkl", "wb") as f:
        pickle.dump(vector_store, f)
    
    return vector_store
# ✅ YouTube API Function (With Error Handling)
def get_youtube_videos(query):
    """Fetches relevant YouTube videos based on the query."""
    if not YOUTUBE_API_KEY:
        return "⚠️ YouTube API key is missing. Unable to fetch videos."

    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        request = youtube.search().list(q=query, part="snippet", maxResults=3, type="video")
        response = request.execute()
        return "\n".join([f"🎥 {video['snippet']['title']} - https://www.youtube.com/watch?v={video['id']['videoId']}" for video in response["items"]])
    except Exception as e:
        return f"⚠️ Error fetching YouTube videos: {e}"

# ✅ Fetch Blog Articles
def get_blog_articles(query):
    """Fetches blog articles from Medium, Dev.to, and TowardsDataScience."""
    results = list(DDGS().text(f"{query} site:medium.com OR site:dev.to OR site:towardsdatascience.com", max_results=5))
    return "\n".join([f"{result['title']} - {result['href']}" for result in results])

# ✅ Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    raw_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw_text += page.extract_text() + "\n"
    return raw_text if raw_text.strip() else "Error: Could not extract text from PDF."

# ✅ Get Vector Store (Thread-Safe)
def get_vector_store():
    try:
        index = faiss.read_index("vector_store.index")
        with open("vector_store.pkl", "rb") as f:
            vector_store = pickle.load(f)
        return vector_store
    except:
        return None  # If no stored vector store, return None


# ✅ API Route: Upload PDF & Initialize Vector Store
@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    raw_text = extract_text_from_pdf(file_path)
    if raw_text.strip():
        text_chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200).split_text(raw_text)
        g.vector_store = initialize_vector_store(text_chunks)
        return jsonify({"message": "PDF processed successfully", "file": filename})
    else:
        return jsonify({"error": "Could not extract text from PDF"}), 400

# ✅ Define External Tools
tools = [
    Tool(name="Wikipedia", func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run, description="Search Wikipedia."),
    Tool(name="ArXiv", func=ArxivQueryRun(api_wrapper=ArxivAPIWrapper()).run, description="Retrieve research papers."),
    Tool(name="Blog Articles", func=get_blog_articles, description="Fetch blog articles related to the topic."),
    Tool(name="YouTube", func=get_youtube_videos, description="Fetch relevant YouTube videos."),
]

# ✅ Initialize LangChain Agent
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GEMINI_API_KEY)
agent_executor = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# ✅ Chatbot Query Processing (With Learning Resources)
def query_with_learning_resources(vector_store, user_query):
    print("\n🔍 Searching in PDF database...")

    retrieved_docs = vector_store.vectorstore.similarity_search(user_query, k=3)
    combined_context = " ".join([doc.page_content for doc in retrieved_docs])
    response_text = ""

    if retrieved_docs:
        print("\n📄 Answer found in PDF database.")
        response_text = llm.invoke([{"role": "user", "content": combined_context + "\n\n" + user_query}])
    else:
        print("\n🌍 No relevant answer found in PDF. Using external sources...")
        response_text = agent_executor.run(user_query)

    # ✅ Fetch Additional Learning Resources
    blog_articles, youtube_videos = None, None
    learning_keywords = ["learn", "study", "resources", "tutorials", "guide", "best way to understand", "how to learn"]
    if any(keyword in user_query.lower() for keyword in learning_keywords):
        print("\n📚 Fetching additional learning materials...")
        blog_articles = get_blog_articles(user_query)
        youtube_videos = get_youtube_videos(user_query)

    # ✅ Return Structured JSON Response
    return {
        "response": response_text,
        "learning_materials": {
            "blogs": blog_articles if blog_articles else "No relevant blog articles found.",
            "youtube": youtube_videos if youtube_videos else "No relevant YouTube videos found."
        }
    }

# ✅ API Route: Chat with AI
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_query = data.get("query", "")

    if "vector_store" not in g or g.vector_store is None:
        return jsonify({"error": "No PDF uploaded yet. Please upload a document first."}), 400

    response = query_with_learning_resources(g.vector_store, user_query)
    return jsonify(response)

# ✅ Run Flask App
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # ✅ Ensure proper port binding for Render
    app.run(host="0.0.0.0", port=port, debug=True)
