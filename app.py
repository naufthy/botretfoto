import os
import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import GeminiEmbeddingFunction
import google.generativeai as genai

# 1. PAGE LAYOUT CONFIGURATION
st.set_page_config(
    page_title="ShutterBuddy: Photography RAG Assistant",
    page_icon="📸",
    layout="wide"
)

# Application title
st.title("📸 ShutterBuddy")
st.markdown("##### *Your friendly neighborhood photography mentor—helping you learn from the ground up!*")

# 2. SIDEBAR FOR CONFIGURATION
st.sidebar.header("⚙️ Configuration")

# API Key input
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))

# Reset chat button
if st.sidebar.button("🗑️ Clear Conversation"):
    st.session_state.messages = []
    st.success("Conversation cleared!")
    st.rerun()

# The Creative Toggle Parameter: Explanation level selector
explanation_level = st.sidebar.radio(
    "💡 Mentor Explanation Style",
    options=["Simplified Mentor Mode", "Detailed Instructor Mode"],
    help="Simplified Mode uses simple everyday analogies and avoids heavy jargon. Detailed Mode provides full technical specifications, f-stops, shutter speeds, and camera physics."
)

st.sidebar.markdown("""
---
### 🛠️ How ShutterBuddy Works:
1. **RAG Vector Search:** Queries our local database containing expert-curated photography guides.
2. **Simplified Mode:** Translates concepts into easy everyday analogies (e.g., aperture = window blinds).
3. **Detailed Mode:** Breaks down exact camera settings (e.g., 1/500s, f/2.8, ISO 100).
""")

# 3. INITIALIZE VECTOR DATABASE & GEMINI CONFIGURATION
def get_db_collection(api_key):
    try:
        # Load persistent ChromaDB client
        client = chromadb.PersistentClient(path="./photography_db")
        
        # Setup Gemini embedding function
        embedding_fn = GeminiEmbeddingFunction(
            model_name="models/embedding-001",
            api_key=api_key
        )
        
        # Get existing collection
        collection = client.get_collection(
            name="photography_guides",
            embedding_function=embedding_fn
        )
        return collection
    except Exception as e:
        st.error(f"Failed to load the vector database. Make sure you ran 'python ingest.py' first. Error: {e}")
        return None

# Check if API Key is set
if not api_key:
    st.info("👈 Please enter your Gemini API Key in the sidebar to start learning!")
    st.stop()

# Configure global Gemini settings
genai.configure(api_key=api_key)

# Load the database collection
collection = get_db_collection(api_key)
if collection is None:
    st.warning("⚠️ Vector Database not detected or empty! Follow Step 2 in the tutorial to ingest photography data.")
    st.stop()

# 4. CHAT STATE INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. CONVERSATIONAL RAG PIPELINE
if prompt := st.chat_input("Ask ShutterBuddy a photography question... (e.g., 'How do I take a portrait with a blurry background?')"):
    # 5.1 Display user's question
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 5.2 RAG Retrieval from ChromaDB
    with st.status("🔍 Searching photography knowledge base...", expanded=False) as status:
        try:
            # Query top 4 most relevant chunks
            results = collection.query(
                query_texts=[prompt],
                n_results=4
            )
            
            # Format retrieved context
            retrieved_docs = results['documents']
            context = "\n\n---\n\n".join(retrieved_docs)
            status.update(label="✓ Knowledge retrieved successfully!", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Error querying database: {e}")
            st.stop()

    # 5.3 Select Dynamic Prompting Instructions based on sidebar toggle
    if explanation_level == "Simplified Mentor Mode":
        system_instruction = (
            "You are ShutterBuddy, an encouraging and friendly photography mentor helping absolute beginners. "
            "Your goal is to explain concepts as simply as possible using everyday, relatable analogies. "
            "For example, compare aperture to window blinds or pupils, shutter speed to a fast-closing shutter door, "
            "and ISO to sunglasses or artificial light amplifiers.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Ground your answers strictly on the retrieved knowledge provided below.\n"
            "2. Avoid using heavy technical terms. If you must use a technical term (like f-stops), explain it immediately in plain English.\n"
            "3. Keep your response highly encouraging, friendly, and structured. Include 1 easy practice tip at the end.\n"
            "4. Do NOT make up any information outside of the retrieved knowledge base. If the retrieved knowledge doesn't discuss the question, politely say so."
        )
    else:
        system_instruction = (
            "You are ShutterBuddy, an expert, professional photography instructor teaching intermediate and advanced students. "
            "Your goal is to provide deep, technically accurate explanations.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Ground your answers strictly on the retrieved knowledge provided below.\n"
            "2. Provide precise, specific camera settings, f-stop calculations, optical mechanics, and step-by-step technical workflows.\n"
            "3. Write in a formal, educational, and precise tone.\n"
            "4. Highlight the exact tradeoffs of exposure settings (Aperture vs. Shutter Speed vs. ISO).\n"
            "5. Do NOT make up any information outside of the retrieved knowledge base. If the retrieved knowledge doesn't discuss the question, politely say so."
        )

    # 5.4 Format Prompt with Context
    final_prompt = f"""
Retrieved Photography Knowledge:
--------------------------------
{context}
--------------------------------

User's Question: {prompt}

Please answer the user's question by applying the photography knowledge retrieved above, and matching the requested mentor style. Do not use external knowledge outside of what is in the retrieved text.
"""

    # 5.5 Generate response using Gemini 2.5 Flash
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking and formatting response..."):
            try:
                model = genai.GenerativeModel(
                    model_name="gemini-2.5-flash",
                    system_instruction=system_instruction
                )
                
                # Send the final prompt to the model
                response = model.generate_content(final_prompt)
                
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"Error generating response from Gemini API: {e}")