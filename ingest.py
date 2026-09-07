import os
import time
import chromadb
from chromadb.utils.embedding_functions import GoogleGeminiEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    # 1. Get Gemini API Key
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("[!] GEMINI_API_KEY environment variable belum diset.")
        print('    Jalankan: $env:GEMINI_API_KEY="YOUR_API_KEY"')
        return

    print("[+] Gemini API key ditemukan.")

    # 2. Read photography_guide.txt
    guide_path = "photography_guide.txt"

    if not os.path.exists(guide_path):
        print(f"[-] Error: '{guide_path}' tidak ditemukan.")
        print("    Pastikan file berada di folder yang sama dengan ingest.py.")
        return

    print("[+] Reading photography guide...")

    with open(guide_path, "r", encoding="utf-8") as f:
        text_content = f.read()

    if not text_content.strip():
        print("[-] Error: photography_guide.txt kosong.")
        return

    # 3. Split document into chunks
    print("[+] Splitting text into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )

    chunks = text_splitter.split_text(text_content)

    print(f"[+] Created {len(chunks)} chunks from the document.")

    if not chunks:
        print("[-] Tidak ada chunk yang berhasil dibuat.")
        return

    # 4. Initialize persistent ChromaDB
    db_path = "./photography_db"

    print(f"[+] Initializing persistent database at: {db_path}")

    client = chromadb.PersistentClient(path=db_path)

    # 5. Gemini embedding function
    print("[+] Initializing Gemini embedding function...")

    embedding_fn = GoogleGeminiEmbeddingFunction(
    model_name="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",
    dimension=768,
)

    # 6. Create / load collection
    print("[+] Creating/retrieving ChromaDB collection...")

    collection = client.get_or_create_collection(
        name="photography_guides",
        embedding_function=embedding_fn,
    )

    print(f"[+] Existing documents in collection: {collection.count()}")

    # 7. Add chunks
    print("[+] Uploading chunks with Gemini embeddings...")

    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i}"

        # Skip existing chunk
        existing = collection.get(ids=[chunk_id])

        if existing["ids"]:
            print(f"    -> Skipping existing chunk {i + 1}/{len(chunks)}")
            continue

        collection.add(
            ids=[chunk_id],
            documents=[chunk],
        )

        print(
            f"    -> Added chunk {i + 1}/{len(chunks)} "
            f"(ID: {chunk_id})"
        )

        # Small delay between API calls
        time.sleep(1)

    print()
    print("[V] SUCCESS!")
    print(
        f"[V] Vector database saved in '{db_path}'."
    )
    print(
        f"[V] Total documents in collection: {collection.count()}"
    )


if __name__ == "__main__":
    main()