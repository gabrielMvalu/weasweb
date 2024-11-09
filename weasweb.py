import streamlit as st
import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pdfplumber
import docx
from pymongo import MongoClient

# Configurare initiala OpenAI
openai.api_key = "YOUR_OPENAI_API_KEY"

# Configurare initiala MongoDB Atlas
client = MongoClient("YOUR_MONGODB_ATLAS_CONNECTION_STRING")
db = client['cv_matching_db']
candidates_collection = db['candidates']
matches_collection = db['matches']

# Functie pentru a crea embedding-uri si a reduce dimensionalitatea folosind OpenAI
@st.cache_data
def create_and_reduce_embedding(text, n_components=50):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embedding = response['data'][0]['embedding']
    pca = PCA(n_components=min(n_components, len(embedding)))
    reduced_embedding = pca.fit_transform(np.array(embedding).reshape(1, -1))
    return reduced_embedding.flatten()

# Functie pentru a extrage textul din PDF
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Functie pentru a extrage textul din Word
def extract_text_from_word(file):
    doc = docx.Document(file)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text

# Streamlit UI
st.title("CV Matching Tool")

# Sectiune pentru incarcare CV-uri
uploaded_files = st.file_uploader("Incarca CV-uri (PDF sau Word)", accept_multiple_files=True)
job_description = st.text_area("Descriere job")

# Lista pentru stocarea embedding-urilor CV-urilor
candidates = []

if uploaded_files and job_description:
    # Procesarea descrierii jobului
    job_embedding = create_and_reduce_embedding(job_description)

    for uploaded_file in uploaded_files:
        # Extrage textul din CV
        if uploaded_file.type == "application/pdf":
            cv_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            cv_text = extract_text_from_word(uploaded_file)
        else:
            st.warning(f"Formatul fisierului {uploaded_file.name} nu este suportat.")
            continue

        # Creare embedding pentru textul extras
        cv_embedding = create_and_reduce_embedding(cv_text)

        # Stocheaza embedding-ul in MongoDB Atlas
        candidates_collection.insert_one({'name': uploaded_file.name, 'embedding': cv_embedding.tolist()})
        
        # Adauga la lista pentru vizualizare
        candidates.append({'name': uploaded_file.name, 'embedding': cv_embedding})

    # Calcularea similaritatii cosine dintre descrierea jobului si embedding-urile CV-urilor
    embeddings = np.array([candidate['embedding'] for candidate in candidates])
    job_embedding = np.array(job_embedding).reshape(1, -1)
    similarities = cosine_similarity(job_embedding, embeddings).flatten()

    # Creare DataFrame pentru a organiza datele
    df = pd.DataFrame({'Candidate': [candidate['name'] for candidate in candidates], 'Similarity': similarities})
    df = df.sort_values(by='Similarity', ascending=False)

    # Afisare tabel si heatmap
    st.write("### Rezultate potrivire")
    st.dataframe(df)

    # Generare heatmap pentru vizualizare
    st.write("### Heatmap Similaritate")
    fig, ax = plt.subplots()
    cax = ax.imshow(similarities.reshape(1, -1), cmap='hot', aspect='auto')
    ax.set_xticks(range(len(candidates)))
    ax.set_xticklabels(df['Candidate'], rotation=90, fontsize=8)
    ax.set_xlabel("Candidates")
    ax.set_ylabel("Similarity Score")
    fig.colorbar(cax, ax=ax)
    st.pyplot(fig)

    # Buton pentru a salva informatiile in MongoDB Atlas
    if st.button("Salveaza informatiile in MongoDB Atlas"):
        for index, row in df.iterrows():
            matches_collection.insert_one({'job_description': job_description, 'candidate_name': row['Candidate'], 'similarity': row['Similarity']})
        st.success("Informatiile au fost salvate in MongoDB Atlas.")
