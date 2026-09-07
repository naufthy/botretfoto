import os
import time
import chromadb
from chromadb.utils.embedding_functions import GeminiEmbeddingFunction
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    # 1. Setup API Key
    api_key = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
    if api_key == "YOUR_GEMINI_API_KEY_HERE" or not api_key:
        print("[!] Warning: Please set GEMINI_API_KEY environment variable or edit the script with your key.")
        api_key = input("Enter your Gemini API Key to proceed: ").strip()
    
    genai.configure(api_key=api_key)
    
    # 2. Read photography_guide.txt
    guide_path = "photography_guide.txt"
    if not os.path.exists(guide_path):
        print(f"[-] Error: '{guide_path}' not found. Please ensure the file is in the same directory.")
        return
        
    print("[+] Reading photography guide...")
    with open(guide_path, "r", encoding="utf-8") as f:
        text_content = f.read()
    
    # 3. Chunking the text using RecursiveCharacterTextSplitter (LangChain)
    # Chunk size: 1000 characters, overlap: 100 characters to prevent loss of context
    print("[+] Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text_content)
    print(f"[+] Created {len(chunks)} chunks from the document.")

    # 4. Initialize persistent ChromaDB client
    db_path = "./photography_db"
    print(f"[+] Initializing persistent database at: {db_path}")
    client = chromadb.PersistentClient(path=db_path)
    
    # 5. Define Gemini Embedding Function
    embedding_fn = GeminiEmbeddingFunction(
        model_name="models/embedding-001",
        api_key=api_key
    )
    
    # Create or load collection
    print("[+] Creating/Retrieving ChromaDB collection 'photography_guides'...")
    collection = client.get_or_create_collection(
        name="photography_guides",
        embedding_function=embedding_fn
    )
    
    # 6. Add chunks to vector database
    print("[+] Uploading chunks with vector embeddings to ChromaDB...")
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i}"
        
        # Add document to collection (ChromaDB auto-embeds using the provided GeminiEmbeddingFunction)
        collection.add(
            documents=[chunk],
            ids=[chunk_id]
        )
        print(f"    -> Added chunk {i+1}/{len(chunks)} (ID: {chunk_id})")
        
        # Introduce a delay of 1.0 second between API calls to stay within free tier limits (60 RPM)
        time.sleep(1.0)
        
    print("\n[V] Success! Your local vector database has been successfully built and saved in './photography_db'!")

if __name__ == "__main__":
    main()