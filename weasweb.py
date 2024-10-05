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
    openai.api_key = api_key

    # Define the JSON schema
    function_schema = {
        "name": "extract_cv_info",
        "description": "Extracts information from a CV and structures it in JSON format.",
        "parameters": {
            "type": "object",
            "properties": {
                "contact_details": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Candidate's full name"},
                        "email": {"type": "string", "description": "Email address"},
                        "phone": {"type": "string", "description": "Phone number"}
                    },
                    "required": ["name"]
                },
                "experience": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "position": {"type": "string", "description": "Job title"},
                            "company": {"type": "string", "description": "Company name"},
                            "start_date": {"type": "string", "description": "Start date"},
                            "end_date": {"type": "string", "description": "End date"}
                        },
                        "required": ["position", "company", "start_date"]
                    }
                },
                "education": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "institution": {"type": "string", "description": "Institution name"},
                            "degree": {"type": "string", "description": "Degree earned"},
                            "start_date": {"type": "string", "description": "Start date"},
                            "end_date": {"type": "string", "description": "End date"}
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

    # Build the messages
    messages = [
        {"role": "system", "content": "You are an assistant that extracts information from CVs and structures it according to a specified JSON schema."},
        {"role": "user", "content": f"Please extract the information from the following CV and return it according to the specified schema.\n\nCV:\n{cv_text}"}
    ]

    try:
        response = openai.Chat.create(
            model="gpt-4-0613",  # Ensure you have access to this model
            messages=messages,
            functions=[function_schema],
            function_call={"name": "extract_cv_info"},
            strict=True,  # Enable Structured Outputs
            max_tokens=1500,
            temperature=0
        )

        # Process the response
        message = response["message"]

        if "function_call" in message:
            extracted_data = json.loads(message["function_call"]["arguments"])
            return extracted_data
        else:
            raise ValueError("The model did not return structured data according to the schema.")

    except openai.exceptions.OpenAIError as e:
        raise Exception(f"An error occurred while accessing OpenAI: {str(e)}")

# Function to populate a Word template with extracted data
def populate_word_template(extracted_data):
    template_path = "cv_template.docx"
    doc = Document(template_path)

    # Replace placeholders in the Word template
    for paragraph in doc.paragraphs:
        if '{{name}}' in paragraph.text:
            paragraph.text = paragraph.text.replace('{{name}}', extracted_data['contact_details']['name'])
        if '{{contact_info}}' in paragraph.text:
            email = extracted_data['contact_details'].get('email', '')
            phone = extracted_data['contact_details'].get('phone', '')
            contact_info = f"Email: {email}, Phone: {phone}"
            paragraph.text = paragraph.text.replace('{{contact_info}}', contact_info)
        if '{{experience}}' in paragraph.text:
            experience_text = ''
            for exp in extracted_data.get('experience', []):
                position = exp.get('position', '')
                company = exp.get('company', '')
                start_date = exp.get('start_date', '')
                end_date = exp.get('end_date', 'Present')
                experience_text += f"{position} at {company} ({start_date} - {end_date})\n"
            paragraph.text = paragraph.text.replace('{{experience}}', experience_text)
        if '{{education}}' in paragraph.text:
            education_text = ''
            for edu in extracted_data.get('education', []):
                degree = edu.get('degree', '')
                institution = edu.get('institution', '')
                start_date = edu.get('start_date', '')
                end_date = edu.get('end_date', 'Present')
                education_text += f"{degree} at {institution} ({start_date} - {end_date})\n"
            paragraph.text = paragraph.text.replace('{{education}}', education_text)
        if '{{skills}}' in paragraph.text:
            skills_text = ', '.join(extracted_data.get('skills', []))
            paragraph.text = paragraph.text.replace('{{skills}}', skills_text)

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
        openai_api_key = st.text_input("Enter your OpenAI API key", type="password")

        # Logout button
        if st.button("Logout"):
            st.session_state.authenticated = False

# Only proceed if the user is authenticated and has provided the OpenAI API key
if st.session_state.authenticated:
    if not openai_api_key:
        st.info("Please enter your OpenAI API key in the sidebar.")
    else:
        # Display the logo in the sidebar
        try:
            logo = Image.open("logo.png")
            st.sidebar.image(logo, use_column_width=True)
        except Exception:
            st.sidebar.write("Logo not found.")

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
                extracted_data = get_structured_data_from_openai(cv_text, openai_api_key)

                # Populate the Word template with the extracted data
                restructured_cv = populate_word_template(extracted_data)

                # Download button for the restructured CV
                st.download_button(
                    label="Download Restructured CV",
                    data=restructured_cv,
                    file_name="restructured_cv.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            except openai.exceptions.OpenAIError as e:
                st.error(f"An error occurred while accessing OpenAI: {str(e)}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

        else:
            st.sidebar.info("Please upload a CV to process.")

