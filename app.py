import streamlit as st
from src.rag.graph_retriever import retrieve_from_graph, format_rag_context
from src.llm.generator import call_llm, refine_query, check_llm_status
from src.rag.graph_db import neo4j_handler
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title='RAG Kesehatan Masyarakat', page_icon='🏥', layout='wide')

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .stButton > button {
        border-radius: 10px;
        background-color: #2e7d32;
        color: white;
    }
    .response-box {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        border-left: 5px solid #2e7d32;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title('🏥 Sistem RAG Kesehatan Masyarakat (Neo4j + Graph RAG)')
st.markdown("""
Selamat datang di chatbot edukasi kesehatan masyarakat. 
Sistem ini menggunakan **Graph RAG** dengan data dari **Neo4j** untuk memberikan jawaban yang akurat berdasarkan referensi buku kesehatan masyarakat.
""")

# Sidebar for history and info
with st.sidebar:
    st.header("Info Sistem")
    
    # Status Koneksi
    st.subheader("Status Layanan")
    
    # Cek Neo4j
    neo_status = neo4j_handler.check_connection()
    if neo_status:
        st.success("✅ Neo4j Connected")
    else:
        st.error("❌ Neo4j Disconnected")
        
    # Cek LLM
    llm_status = check_llm_status()
    if llm_status:
        st.success("✅ LLM Provider Active")
    else:
        st.error("❌ LLM Provider Down")
        
    st.divider()
    
    # Fitur Buka Neo4j
    st.subheader("Tools")
    neo4j_url = os.getenv('NEO4J_URI', '').strip("'\"")
    
    # Deteksi jika menggunakan Neo4j Aura
    if "databases.neo4j.io" in neo4j_url:
        browser_url = "https://console.neo4j.io/"
        btn_label = "🌐 Buka Neo4j Aura Console"
    else:
        # Default untuk local/self-hosted
        browser_url = neo4j_url.replace('bolt://', 'http://').replace('neo4j+s://', 'https://').replace('neo4j://', 'http://')
        if ":7687" in browser_url:
            browser_url = browser_url.replace(":7687", ":7474")
        btn_label = "🌐 Buka Neo4j Browser"

    if st.button(btn_label):
        st.link_button("Klik untuk Membuka", browser_url)
    
    st.divider()
    
    st.info("""
    - **Database**: Neo4j Graph Database
    - **Retriever**: Hybrid Search (Vector + BM25)
    - **Optimization**: NumPy Vectorized + HyDE (Conditional)
    - **LLM**: Llama-3.3-70b-versatile (Groq)
    """)
    
    if st.button("Hapus Riwayat"):
        st.session_state.messages = []
        st.rerun()

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Apa yang ingin Anda tanyakan seputar kesehatan masyarakat?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mencari referensi dan menyusun jawaban..."):
            # 0. Query Refinement (for follow-up questions)
            # Use history (excluding the current prompt just added)
            refined_q = refine_query(prompt, st.session_state.messages[:-1])
            if refined_q != prompt:
                st.caption(f"🔍 Mencari tentang: *{refined_q}*")

            # 1. Retrieval using refined query - Increase top_k for better context
            chunks = retrieve_from_graph(refined_q, top_k=5)
            
            # Check if chunks exist and have a minimum score threshold
            if not chunks or (len(chunks) > 0 and chunks[0].get('hybrid_score', 0) < 0.15):
                st.warning("Maaf, saya tidak menemukan informasi yang cukup spesifik dalam database buku referensi saya untuk menjawab pertanyaan ini.")
                st.session_state.messages.append({"role": "assistant", "content": "Maaf, saya tidak menemukan informasi yang relevan dalam database saya."})
            else:
                rag_ctx = format_rag_context(chunks)
                
                # 3. LLM Call - PARAGRAPH FORMAT, NO CITATIONS
                SYS = (
                    "ANDA ADALAH ASISTEN EDUKASI KESEHATAN MASYARAKAT YANG RAMAH DAN MUDAH DIPAHAMI.\n"
                    "TUGAS: Menjelaskan topik kesehatan HANYA berdasarkan DATA REFERENSI yang disediakan.\n\n"
                    "ATURAN PENULISAN:\n"
                    "1. JAWAB DALAM BENTUK PARAGRAPH yang mengalir secara alami (bukan poin-poin).\n"
                    "2. Gunakan bahasa yang sederhana, hangat, dan mudah dimengerti oleh masyarakat awam.\n"
                    "3. DILARANG KERAS mencantumkan sumber, nomor halaman, kode referensi, atau sitasi apa pun.\n"
                    "4. JANGAN gunakan pengetahuan di luar DATA REFERENSI.\n"
                    "5. Jika data tidak ada, katakan dengan sopan bahwa informasi tersebut belum tersedia di database.\n\n"
                    "Fokus pada penjelasan yang edukatif dan menenangkan."
                )

                USR = (
                    f"--- DATA REFERENSI ---\n{rag_ctx}\n----------------------\n\n"
                    f"PERTANYAAN: {prompt}\n\n"
                    "Jawablah dalam beberapa paragraf yang rapi dan mudah dipahami tanpa menyebutkan sumber datanya."
                )
                
                # Use history if available (limit to last 4 messages for context)
                msgs = [{"role": "system", "content": SYS}]
                # Add last 4 messages from history
                for msg in st.session_state.messages[-5:-1]:
                    msgs.append({"role": msg["role"], "content": msg["content"]})
                msgs.append({"role": "user", "content": USR})
                
                response = call_llm(msgs)
                
                st.markdown(response)
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response})

# Examples
if not st.session_state.messages:
    st.markdown("### Contoh Pertanyaan:")
    cols = st.columns(2)
    examples = [
        "Apa itu stunting dan bagaimana cara mencegahnya?",
        "Bagaimana cara mencegah demam berdarah?",
        "Apa itu PHBS dan penerapannya?",
        "Jelaskan epidemiologi tuberkulosis"
    ]
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex):
            # This is a bit tricky in Streamlit to trigger chat_input, 
            # so we just suggest it or the user can copy-paste.
            st.info(f"Silakan salin pertanyaan ini ke chat box: `{ex}`")
