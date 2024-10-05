import streamlit as st
import pdfplumber
import docx
from io import BytesIO
from docx import Document
import openai
import json
from PIL import Image

# Access secrets (username and password from Streamlit Secrets)
USERNAME = st.secrets["credentials"]["USERNAME"]
PASSWORD = st.secrets["credentials"]["PASSWORD"]

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
def get_structured_data_from_openai(cv_text, api_key):
    import openai
    openai.api_key = api_key

    # Definește schema JSON pe care dorești ca modelul să o respecte
    function_schema = {
        "name": "extract_cv_info",
        "description": "Extrage informații dintr-un CV și le structurează în format JSON.",
        "parameters": {
            "type": "object",
            "properties": {
                "contact_details": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Numele complet al candidatului"},
                        "email": {"type": "string", "description": "Adresa de email"},
                        "phone": {"type": "string", "description": "Numărul de telefon"}
                    },
                    "required": ["name"]
                },
                "experience": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "position": {"type": "string", "description": "Titlul poziției"},
                            "company": {"type": "string", "description": "Numele companiei"},
                            "start_date": {"type": "string", "description": "Data de început"},
                            "end_date": {"type": "string", "description": "Data de finalizare"}
                        },
                        "required": ["position", "company", "start_date"]
                    }
                },
                "education": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "institution": {"type": "string", "description": "Numele instituției"},
                            "degree": {"type": "string", "description": "Titlul obținut"},
                            "start_date": {"type": "string", "description": "Data de început"},
                            "end_date": {"type": "string", "description": "Data de finalizare"}
                        },
                        "required": ["institution", "degree", "start_date"]
                    }
                },
                "skills": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["contact_details"]
        }
    }

    # Construiește mesajele pentru model
    messages = [
        {"role": "system", "content": "Ești un asistent care extrage informații din CV-uri și le structurează conform unei scheme JSON specificate."},
        {"role": "user", "content": f"Te rog să extragi informațiile din următorul CV și să le returnezi conform schemei specificate.\n\nCV:\n{cv_text}"}
    ]

    # Apelează API-ul OpenAI cu parametrul strict: True
    response = openai.ChatCompletion.create(
        model="gpt-4o-2023-10-06",  # Asigură-te că ai acces la acest model
        messages=messages,
        functions=[function_schema],
        function_call={"name": "extract_cv_info"},
        strict=True,  # Activează Structured Outputs
        max_tokens=1500,
        temperature=0
    )

    # Verifică dacă modelul a returnat un apel de funcție
    message = response["choices"][0]["message"]

    if message.get("function_call"):
        # Extrage argumentele funcției returnate de model
        extracted_data = json.loads(message["function_call"]["arguments"])
        return extracted_data
    else:
        raise ValueError("Modelul nu a returnat date structurate conform schemei.")


# Function to populate a Word template with extracted data
def populate_word_template(extracted_data):
    template_path = "cv_template.docx"
    doc = Document(template_path)

    # Replace placeholders in the Word template
    for paragraph in doc.paragraphs:
        if '{{name}}' in paragraph.text:
            paragraph.text = paragraph.text.replace('{{name}}', extracted_data['contact_details']['name'])
        if '{{contact_info}}' in paragraph.text:
            contact_info = f"Email: {extracted_data['contact_details']['email']}, Phone: {extracted_data['contact_details']['phone']}"
            paragraph.text = paragraph.text.replace('{{contact_info}}', contact_info)
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

# Sidebar for login and OpenAI API key
with st.sidebar:
    st.title("Login")
    if not st.session_state.authenticated:
        input_username = st.text_input("Username", key="username")
        input_password = st.text_input("Password", type="password", key="password")

        if st.button("Login"):
            # Using credentials from st.secrets for authentication
            if input_username == USERNAME and input_password == PASSWORD:
                st.session_state.authenticated = True
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password.")

    if st.session_state.authenticated:
        # Input for OpenAI API key from user
        openai_api_key = st.text_input("Introduceți cheia de acces OpenAI", type="password")

        # Logout button
        if st.button("Logout"):
            st.session_state.authenticated = False

# Only proceed if the user is authenticated and has provided the OpenAI API key
if st.session_state.authenticated:
    if not openai_api_key:
        st.info("Vă rugăm să introduceți cheia de acces OpenAI în bara laterală.")
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
            try:
                structured_data = get_structured_data_from_openai(cv_text, openai_api_key)

                # Convert the JSON response to a Python dictionary
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
            except openai.error.OpenAIError as e:
                st.error(f"A apărut o eroare la accesarea OpenAI: {str(e)}")

        else:
            st.sidebar.info("Please upload a CV to process.")


