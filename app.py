import os
import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import GoogleGeminiEmbeddingFunction
from google import genai

# 1. PAGE CONFIGURATION

st.set_page_config(
    page_title="BotretFoto: Photography RAG Assistant",
    page_icon="📸",
    layout="wide"
)

st.title("📸 BotretFoto")
st.markdown(
    "##### *Your friendly neighborhood photography mentor—"
    "helping you learn from the ground up!*"
)

# 2. SIDEBAR
st.sidebar.header("⚙️ Configuration")

api_key = st.sidebar.text_input(
    "Enter Gemini API Key",
    type="password",
    value=os.environ.get("GEMINI_API_KEY", "")
)

if st.sidebar.button("🗑️ Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

explanation_level = st.sidebar.radio(
    "💡 Mentor Explanation Style",
    options=[
        "Simplified Mentor Mode",
        "Detailed Instructor Mode"
    ],
    help=(
        "Simplified Mode uses simple everyday analogies. "
        "Detailed Mode provides technical photography explanations."
    )
)

st.sidebar.markdown("""
---
### 🛠️ How BotretFoto Works:

1. **RAG Vector Search:** Queries our local photography knowledge database.
2. **Simplified Mode:** Explains photography using simple analogies.
3. **Detailed Mode:** Explains camera settings and technical concepts.
""")


# 3. CHECK API KEY
if not api_key:
    st.info(
        "👈 Please enter your Gemini API Key in the sidebar "
        "to start learning!"
    )
    st.stop()


# 4. GEMINI CLIENT
try:
    # Google GenAI SDK reads the API key explicitly here.
    gemini_client = genai.Client(api_key=api_key)

except Exception as e:
    st.error(f"Failed to initialize Gemini client: {e}")
    st.stop()



# 5. LOAD CHROMA VECTOR DATABASE
@st.cache_resource
def get_db_collection():
    try:
        db_path = "./photography_db"

        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Database folder '{db_path}' does not exist."
            )

        client = chromadb.PersistentClient(
            path=db_path
        )

        embedding_fn = GoogleGeminiEmbeddingFunction(
            model_name="gemini-embedding-001",
            task_type="RETRIEVAL_QUERY",
            dimension=768,
            api_key_env_var="GEMINI_API_KEY"
        )

        collection = client.get_collection(
            name="photography_guides",
            embedding_function=embedding_fn
        )

        return collection

    except Exception as e:
        st.error(
            "Failed to load the vector database. "
            "Make sure you ran 'python ingest.py' first.\n\n"
            f"Error: {e}"
        )
        return None


collection = get_db_collection()

if collection is None:
    st.warning(
        "⚠️ Vector Database not detected or empty! "
        "Run `python ingest.py` first."
    )
    st.stop()

if collection.count() == 0:
    st.warning(
        "⚠️ Vector database exists, but it contains no documents."
    )
    st.stop()

# Show small status in sidebar
st.sidebar.success(
    f"📚 Knowledge base: {collection.count()} chunks"
)


# 6. CHAT STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 7. CHAT INPUT
if prompt := st.chat_input(
    "Ask BotretFoto a photography question..."
):
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # RAG SEARCH
    with st.status(
        "🔍 Searching photography knowledge base...",
        expanded=False
    ) as status:

        try:
            results = collection.query(
                query_texts=[prompt],
                n_results=min(4, collection.count())
            )

            documents = results.get("documents", [])

            if not documents or not documents[0]:
                context = "No relevant photography knowledge was found."

            else:
                context = "\n\n---\n\n".join(
                    documents[0]
                )

            status.update(
                label="Knowledge retrieved successfully!",
                state="complete",
                expanded=False
            )

        except Exception as e:
            status.update(
                label="❌ Failed to search knowledge base",
                state="error"
            )

            st.error(
                f"Error querying database: {e}"
            )
            st.stop()

    
    # 8. SYSTEM INSTRUCTION
    if explanation_level == "Simplified Mentor Mode":

        system_instruction = """
You are BotretFoto, a friendly and encouraging photography mentor.

Your audience consists of beginner photographers.

Explain photography concepts using simple everyday analogies.

For example:
- aperture can be compared to window blinds
- shutter speed can be compared to how quickly a door opens/closes
- ISO can be explained as the camera's sensitivity/amplification

IMPORTANT RULES:

1. Ground your answer strictly in the retrieved photography
   knowledge provided by the application.

2. Avoid unnecessary technical jargon.

3. If you use a technical photography term, explain it immediately in simple language.

4. Give practical and actionable advice.

5. End with one simple photography practice tip.

6. Do not invent facts that are not supported by the retrieved knowledge.

7. If the retrieved knowledge does not contain enough information to answer the question, say so honestly.
"""

    else:

        system_instruction = """
You are BotretFoto, an expert professional photography instructor.

Your audience consists of intermediate and advanced photographers.

Provide technically accurate photography explanations.

IMPORTANT RULES:

1. Ground your answer strictly in the retrieved photography knowledge provided by the application.

2. Explain relevant relationships between:
   - aperture
   - shutter speed
   - ISO
   - exposure
   - depth of field
   - motion blur
   - focal length
   - optics

3. Provide precise settings when the retrieved knowledge supports them.

4. Explain tradeoffs between camera settings.

5. Use a formal and educational tone.

6. Do not invent information outside the retrieved knowledge.

7. If the retrieved knowledge does not contain enough information to answer the question, say so honestly.
"""

    
    # 9. FINAL RAG PROMPT
    final_prompt = f"""
Retrieved Photography Knowledge
================================

{context}

================================

User Question:
{prompt}

================================

Instructions:

Answer the user's question using the retrieved photography
knowledge above.

Do not rely on outside information.

Follow the selected mentor explanation style.
"""

    
    # 10. GENERATE GEMINI RESPONSE
    with st.chat_message("assistant"):

        with st.spinner("🧠 Thinking..."):

            try:

                response = gemini_client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=final_prompt,
                    config={
                        "system_instruction": system_instruction
                    }
                )

                answer = response.text

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(
                    f"Error generating response from Gemini API: {e}"
                )