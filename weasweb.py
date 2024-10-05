import streamlit as st
import pdfplumber
import docx
from io import BytesIO
from docx import Document
import openai
from PIL import Image

# Access secrets
USERNAME = st.secrets["credentials"]["USERNAME"]
PASSWORD = st.secrets["credentials"]["PASSWORD"]
OPENAI_API_KEY = st.secrets["credentials"]["OPENAI_API_KEY"]


# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Function to extract text from PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        pages = [page.extract_text() for page in pdf.pages]
    return "\n".join(pages)

# Function to extract text from Word
def extract_text_from_docx(file):
    doc = docx.Document(file)
    fullText = [para.text for para in doc.paragraphs]
    return '\n'.join(fullText)

# Function to get structured data from OpenAI
def get_structured_data_from_openai(cv_text):
    prompt = f"""
    Extract the relevant information from the following CV and structure it in the following JSON format:
    {{
        "name": "Nume Candidat",
        "contact_info": "Info Contact",
        "experience": [
            {{
                "position": "Titlul Jobului",
                "company": "Compania",
                "duration": "Durata",
                "description": "Descriere"
            }}
        ],
        "education": [
            {{
                "degree": "Titlul Diplomei",
                "institution": "Instituția",
                "year": "Anul"
            }}
        ],
        "skills": ["Skill 1", "Skill 2"]
    }}
    CV:
    {cv_text}
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500,
        temperature=0.5
    )
    return response.choices[0].text.strip()

# Function to populate a Word template with extracted data
def populate_word_template(extracted_data):
    template_path = "cv_template.docx"
    doc = Document(template_path)

    # Replace placeholders in the Word template
    for paragraph in doc.paragraphs:
        if '{{name}}' in paragraph.text:
            paragraph.text = paragraph.text.replace('{{name}}', extracted_data['name'])
        if '{{contact_info}}' in paragraph.text:
            paragraph.text = paragraph.text.replace('{{contact_info}}', extracted_data['contact_info'])
        # Add similar logic for experience, education, and skills

    # Save the document to a BytesIO object for download
    output = BytesIO()
    doc.save(output)
    output.seek(0)
    return output

# Streamlit App
st.title("CV Restructuring Tool")

# Initialize session state for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Sidebar for login
if not st.session_state.authenticated:
    st.sidebar.title("Login")
    input_username = st.sidebar.text_input("Username")
    input_password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if input_username == USERNAME and input_password == PASSWORD:
            st.session_state.authenticated = True
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password.")
else:
    # Display the logo in the sidebar
    logo = Image.open("logo.png")
    st.sidebar.image(logo, use_column_width=True)

    # Sidebar for file upload
    st.sidebar.header("Upload CV")
    uploaded_cv = st.sidebar.file_uploader("Upload your CV in PDF or Word format", type=["pdf", "docx"])

    if uploaded_cv is not None:
        if uploaded_cv.type == "application/pdf":
            st.write("Processing PDF...")
            cv_text = extract_text_from_pdf(uploaded_cv)
        elif uploaded_cv.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            st.write("Processing Word Document...")
            cv_text = extract_text_from_docx(uploaded_cv)

        # Get structured data from OpenAI
        structured_data = get_structured_data_from_openai(cv_text)

        # Convert the JSON response to a Python dictionary
        import json
        extracted_data = json.loads(structured_data)

        # Populate the Word template with the extracted data
        restructured_cv = populate_word_template(extracted_data)

        # Download button for the restructured CV
        st.download_button(
            label="Download Restructured CV",
            data=restructured_cv,
            file_name="restructured_cv.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    else:
        st.sidebar.info("Please upload a CV to process.")

    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False


