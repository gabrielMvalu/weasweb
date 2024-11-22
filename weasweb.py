import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import pinecone
import tempfile
import os
import pandas as pd
import plotly.figure_factory as ff
import json

# Configurare Streamlit
st.set_page_config(page_title="CV Matching System", layout="wide")

# Constante
METADATA_PATH = "cv_metadata.json"
INDEX_NAME = "cv-matching-index"

# Initializare Pinecone - Corectat
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
if PINECONE_API_KEY:
    pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")
    pc_client = pinecone
else:
    st.error("PINECONE_API_KEY nu este setat în secrets!")
    pc_client = None

# Functii pentru managementul metadata
def load_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            return json.load(f)
    return {'processed_cvs': []}

def save_metadata(metadata):
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f)

# Initializare variabile sesiune - Corectat
if 'vector_store' not in st.session_state:
    if pc_client and INDEX_NAME in pc_client.list_indexes():
        st.session_state.vector_store = pc_client.Index(INDEX_NAME)
        metadata = load_metadata()
        st.session_state.processed_cvs = metadata.get('processed_cvs', [])
    else:
        st.session_state.vector_store = None
        st.session_state.processed_cvs = []

def process_document(file):
    """Procesează un document PDF sau Word și returnează documentele rezultate."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.name[file.name.rfind('.'):]) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name

    try:
        if file.name.lower().endswith('.pdf'):
            loader = PyPDFLoader(tmp_path)
        elif file.name.lower().endswith(('.docx', '.doc')):
            loader = Docx2txtLoader(tmp_path)
        else:
            os.unlink(tmp_path)
            return None

        documents = loader.load()
        for doc in documents:
            doc.metadata['source'] = file.name
        
        return documents
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def create_or_update_vector_store(documents, existing_store=None):
    """Creează sau actualizează vector store-ul cu documentele procesate."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API Key nu este setat!")
        return None
        
    embeddings = OpenAIEmbeddings()
    vectors = [embeddings.embed_documents([text.content])[0] for text in texts]
    
    if INDEX_NAME not in pc_client.list_indexes():
        pc_client.create_index(INDEX_NAME, dimension=len(vectors[0]))
    
    index = pc_client.Index(INDEX_NAME)
    
    # Upsert cu batch processing pentru eficiență
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_vectors = vectors[i:i + batch_size]
        records = [
            (f"{i+j}", vector, {"source": text.metadata["source"]})
            for j, (text, vector) in enumerate(zip(batch_texts, batch_vectors))
        ]
        index.upsert(vectors=records)
    
    return index

def analyze_job_match(job_description, vector_store):
    """Analizează potrivirea între descrierea jobului și CV-urile stocate."""
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API Key nu este setat!")
        return None
        
    embeddings = OpenAIEmbeddings()
    job_embedding = embeddings.embed_documents([job_description])[0]
    
    query_result = vector_store.query(
        vector=job_embedding,
        top_k=10,
        include_metadata=True
    )
    
    results = []
    seen_cvs = set()
    
    for match in query_result.matches:
        cv_name = match.metadata["source"]
        if cv_name not in seen_cvs:
            results.append({
                'CV': cv_name,
                'Score': match.score
            })
            seen_cvs.add(cv_name)
    
    return pd.DataFrame(results)

def main():
    st.title("🎯 Sistem Avansat de Potrivire CV-uri")
    
    with st.sidebar:
        st.header("Configurare")
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        if st.button("❌ Șterge toate datele"):
            if pc_client and INDEX_NAME in pc_client.list_indexes():
                pc_client.delete_index(INDEX_NAME)
            if os.path.exists(METADATA_PATH):
                os.remove(METADATA_PATH)
            st.session_state.vector_store = None
            st.session_state.processed_cvs = []
            st.success("Toate datele au fost șterse!")
            st.experimental_rerun()

    tab1, tab2, tab3 = st.tabs(["📥 Încărcare CV-uri", "🔍 Potrivire Job", "📊 Management Date"])
    
    with tab1:
        st.header("Încărcare și Procesare CV-uri")
        
        if st.session_state.processed_cvs:
            st.info(f"CV-uri deja procesate: {len(st.session_state.processed_cvs)}")
            if st.checkbox("Arată lista CV-urilor procesate"):
                st.write(st.session_state.processed_cvs)
        
        uploaded_files = st.file_uploader(
            "Încarcă CV-uri noi (PDF sau Word)",
            accept_multiple_files=True,
            type=['pdf', 'docx']
        )
        
        if uploaded_files and st.button("Procesează CV-uri noi"):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("Te rog setează OpenAI API Key în sidebar!")
                return
                
            with st.spinner("Procesare CV-uri în curs..."):
                all_documents = []
                new_cvs = []
                
                for file in uploaded_files:
                    if file.name not in st.session_state.processed_cvs:
                        docs = process_document(file)
                        if docs:
                            all_documents.extend(docs)
                            new_cvs.append(file.name)
                
                if all_documents:
                    vector_store = create_or_update_vector_store(
                        all_documents,
                        st.session_state.vector_store
                    )
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.session_state.processed_cvs.extend(new_cvs)
                        save_metadata({'processed_cvs': st.session_state.processed_cvs})
                        st.success(f"✅ {len(new_cvs)} CV-uri noi procesate cu succes!")
                else:
                    st.warning("Nu există CV-uri noi de procesat.")
    
    with tab2:
        st.header("Potrivire Job cu CV-uri")
        
        if not st.session_state.vector_store:
            st.warning("⚠️ Nu există CV-uri procesate. Te rog încarcă CV-uri în tab-ul anterior.")
            return
        
        job_description = st.text_area(
            "Descriere Job",
            height=200,
            placeholder="Introdu descrierea completă a jobului aici..."
        )
        
        if job_description and st.button("Analizează Potrivirea"):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("Te rog setează OpenAI API Key în sidebar!")
                return
                
            with st.spinner("Analiză în curs..."):
                results_df = analyze_job_match(job_description, st.session_state.vector_store)
                if results_df is not None and not results_df.empty:
                    st.subheader("Rezultate Potrivire")
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        st.dataframe(
                            results_df.style.format({'Score': '{:.2%}'})
                            .background_gradient(cmap='RdYlGn', subset=['Score'])
                        )
                    
                    with col2:
                        fig = ff.create_annotated_heatmap(
                            z=[results_df['Score'].values],
                            x=results_df['CV'].values,
                            y=['Match'],
                            colorscale='RdYlGn',
                            showscale=True
                        )
                        fig.update_layout(
                            title='Heatmap Potrivire CV-uri',
                            xaxis_title='CV-uri',
                            yaxis_title='Nivel Potrivire'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Descarcă Rezultate CSV",
                        data=csv,
                        file_name="rezultate_potrivire.csv",
                        mime="text/csv"
                    )
    
    with tab3:
        st.header("Management Date")
        
        st.subheader("📊 Statistici")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total CV-uri procesate", len(st.session_state.processed_cvs))
        with col2:
            st.metric("Dimensiune Vector Store", "N/A")
        
        st.subheader("💾 Backup & Restore")
        st.info("Datele sunt stocate în Pinecone. Folosiți consola Pinecone pentru management avansat.")

if __name__ == "__main__":
    main()

