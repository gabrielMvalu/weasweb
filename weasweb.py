import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import tempfile
import os
import pandas as pd
import plotly.figure_factory as ff
import numpy as np
import shutil
import json

# Configurare Streamlit
st.set_page_config(page_title="CV Matching System", layout="wide")

# Constante
VECTOR_STORE_PATH = "faiss_index"
METADATA_PATH = "cv_metadata.json"

# Functie pentru incarcarea metadata
def load_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            return json.load(f)
    return {'processed_cvs': []}

# Functie pentru salvarea metadata
def save_metadata(metadata):
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f)

# Initializare variabile sesiune
if 'vector_store' not in st.session_state:
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            st.session_state.vector_store = FAISS.load_local(VECTOR_STORE_PATH, OpenAIEmbeddings())
            metadata = load_metadata()
            st.session_state.processed_cvs = metadata['processed_cvs']
        except Exception as e:
            st.session_state.vector_store = None
            st.session_state.processed_cvs = []
    else:
        st.session_state.vector_store = None
        st.session_state.processed_cvs = []

def process_document(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.name[file.name.rfind('.'):]) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name

    try:
        if file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_path)
        elif file.name.endswith(('.docx', '.doc')):
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

def merge_vector_stores(existing_store, new_store):
    # Merge the indexes
    existing_store.merge_from(new_store)
    return existing_store

def create_or_update_vector_store(documents, existing_store=None):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    new_store = FAISS.from_documents(texts, embeddings)
    
    if existing_store:
        final_store = merge_vector_stores(existing_store, new_store)
    else:
        final_store = new_store
    
    # Salvare vector store
    final_store.save_local(VECTOR_STORE_PATH)
    
    return final_store

def analyze_job_match(job_description, vector_store):
    embeddings = OpenAIEmbeddings()
    job_embedding = embeddings.embed_query(job_description)
    
    similar_docs = vector_store.similarity_search_with_score_by_vector(
        job_embedding,
        k=10
    )
    
    results = []
    seen_cvs = set()  # Pentru a evita duplicate
    
    for doc, score in similar_docs:
        cv_name = doc.metadata['source']
        if cv_name not in seen_cvs:
            similarity_score = 1 - score
            results.append({
                'CV': cv_name,
                'Score': similarity_score
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
        
        # Optiuni pentru resetare si backup
        if st.button("❌ Șterge toate datele"):
            if os.path.exists(VECTOR_STORE_PATH):
                shutil.rmtree(VECTOR_STORE_PATH)
            if os.path.exists(METADATA_PATH):
                os.remove(METADATA_PATH)
            st.session_state.vector_store = None
            st.session_state.processed_cvs = []
            st.success("Toate datele au fost șterse!")
            st.experimental_rerun()

    tab1, tab2, tab3 = st.tabs(["📥 Încărcare CV-uri", "🔍 Potrivire Job", "📊 Management Date"])
    
    with tab1:
        st.header("Încărcare și Procesare CV-uri")
        
        # Afișare CV-uri existente
        if st.session_state.processed_cvs:
            st.info(f"CV-uri deja procesate: {len(st.session_state.processed_cvs)}")
            if st.checkbox("Arată lista CV-urilor procesate"):
                st.write(st.session_state.processed_cvs)
        
        uploaded_files = st.file_uploader(
            "Încarcă CV-uri noi (PDF sau Word)",
            accept_multiple_files=True,
            type=['pdf', 'docx']
        )
        
        if uploaded_files:
            if st.button("Procesează CV-uri noi"):
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
                        # Update sau creare vector store
                        st.session_state.vector_store = create_or_update_vector_store(
                            all_documents,
                            st.session_state.vector_store
                        )
                        
                        # Update metadata
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
            with st.spinner("Analiză în curs..."):
                results_df = analyze_job_match(job_description, st.session_state.vector_store)
                
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
        
        # Statistici
        st.subheader("📊 Statistici")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total CV-uri procesate", len(st.session_state.processed_cvs))
        with col2:
            store_size = 0
            if os.path.exists(VECTOR_STORE_PATH):
                store_size = sum(os.path.getsize(os.path.join(VECTOR_STORE_PATH, f)) 
                               for f in os.listdir(VECTOR_STORE_PATH)) / (1024*1024)
            st.metric("Dimensiune Vector Store", f"{store_size:.2f} MB")
        
        # Backup/Restore
        st.subheader("💾 Backup & Restore")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Creează Backup"):
                if os.path.exists(VECTOR_STORE_PATH):
                    backup_name = f"backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                    shutil.make_archive(backup_name, 'zip', VECTOR_STORE_PATH)
                    with open(f"{backup_name}.zip", "rb") as f:
                        st.download_button(
                            label="📥 Descarcă Backup",
                            data=f,
                            file_name=f"{backup_name}.zip",
                            mime="application/zip"
                        )
        
        with col2:
            uploaded_backup = st.file_uploader("📥 Restaurează din Backup", type=['zip'])
            if uploaded_backup:
                if st.button("Restaurează"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                        tmp_file.write(uploaded_backup.getvalue())
                        backup_path = tmp_file.name
                    
                    # Șterge vector store existent
                    if os.path.exists(VECTOR_STORE_PATH):
                        shutil.rmtree(VECTOR_STORE_PATH)
                    
                    # Extrage backup
                    shutil.unpack_archive(backup_path, VECTOR_STORE_PATH, 'zip')
                    os.unlink(backup_path)
                    
                    # Reîncarcă vector store
                    st.session_state.vector_store = FAISS.load_local(
                        VECTOR_STORE_PATH,
                        OpenAIEmbeddings()
                    )
                    st.success("✅ Backup restaurat cu succes!")
                    st.experimental_rerun()

if __name__ == "__main__":
    main()
