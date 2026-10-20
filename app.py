import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL")

# Capstone Project 14

if not api_key:
    raise ValueError("OPENAI_API_KEY is not configured in the .env file.")
if not model_name:
    raise ValueError("OPENAI_MODEL is not configured in the .env file.")

st.set_page_config(
    page_title="Resume Evaluation System",
    page_icon="📄",
)
st.title("Resume Evaluation System")
st.write(
"Upload a resume and a job description to evaluate "
"the candidate's alignment with the job requirements."
)

resume_file = st.file_uploader(
"Upload Resume PDF",
type=["pdf"]
)
job_description_file = st.file_uploader(
"Upload Job Description PDF",
type=["pdf"]
)

# CCapstone Project 15

llm = ChatOpenAI(
model=model_name,
api_key=api_key
)

resume_prompt = ChatPromptTemplate.from_messages([
(
"system",
"You evaluate resumes against job descriptions."
),
(
"human",
"""
Evaluate the resume against the job description.
Resume:
{resume_text}
Job Description:
{job_description}
Evaluate the resume using the following criteria:
1. Relevant technical skills.
2. Alignment with the stated job requirements.
3. Relevant experience and projects.
4. Matching qualifications.
5. Missing or insufficiently demonstrated requirement
s.
Base the evaluation only on the information
provided in the resume and job description.
Do not invent candidate qualifications.

Capstone Project 16

Do not treat information omitted from the resume
as definitive proof that the candidate lacks
a qualification.
Provide a clear and structured evaluation.
"""
)
])

resume_chain = resume_prompt | llm

if resume_file and job_description_file:
    if st.button("Evaluate Resume"):
        with st.spinner("Evaluating resume..."):
            with open("resume.pdf", "wb") as file:
                file.write(resume_file.getbuffer())
            with open("job_description.pdf", "wb") as file:
                file.write(job_description_file.getbuffer())
            resume_loader = PyPDFLoader("resume.pdf")
            resume_documents = resume_loader.load()

            job_description_loader = PyPDFLoader("job_description.pdf")
            job_description_documents = job_description_loader.load()

            resume_text = "\n".join(
                document.page_content for document in resume_documents
            )
            job_description_text = "\n".join(
                document.page_content for document in job_description_documents
            )

            if not resume_text.strip():
                st.error("No extractable text was found in the resume PDF.")
                st.stop()

            if not job_description_text.strip():
                st.error("No extractable text was found in the job description PDF.")
                st.stop()

            response = resume_chain.invoke({
                "resume_text": resume_text,
                "job_description": job_description_text,
            })

            st.subheader("Resume Evaluation")
            st.write(response.content)
else:
    st.info(
        "Please upload both the resume PDF and "
        "job description PDF."
    )
