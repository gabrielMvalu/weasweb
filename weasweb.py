import streamlit as st
import pdfplumber
import docx
from io import BytesIO
from docx import Document
from PIL import Image

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

# Function to populate a Word template with extracted data
def populate_word_template(extracted_data):
    template_path = "cv_template.docx"  # This should point to your template
    doc = Document(template_path)

    # Replace placeholders in the Word template
    for paragraph in doc.paragraphs:
        if '{{name}}' in paragraph.text:
            paragraph.text = paragraph.text.replace('{{name}}', extracted_data['name'])
        if '{{contact_info}}' in paragraph.text:
            paragraph.text = paragraph.text.replace('{{contact_info}}', extracted_data['contact_info'])
        if '{{experience}}' in paragraph.text:
            paragraph.text = paragraph.text.replace('{{experience}}', extracted_data['experience'])
        if '{{skills}}' in paragraph.text:
            paragraph.text = paragraph.text.replace('{{skills}}', extracted_data['skills'])

    # Save the document to a BytesIO object for download
    output = BytesIO()
    doc.save(output)
    output.seek(0)
    return output

# Streamlit App
st.title("CV Restructuring Tool")

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
    
    # Mock extracted data for now (you can expand this part to do actual extraction)
    extracted_data = {
        "name": "John Doe",
        "contact_info": "johndoe@example.com",
        "experience": "3 years at XYZ Corp",
        "skills": "Python, Machine Learning"
    }

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
