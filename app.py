import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load API key from environment
load_dotenv()
myapi = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=myapi)

def get_text_from_pdf(uploaded_files):
    text = ''
    for pdf in uploaded_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_chunks_from_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Adjust chunk size as needed
        chunk_overlap=200  # Adjust overlap as needed
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_embeddings_from_chunks(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n{context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def get_user_input():
    user_question = st.text_input("💬 Ask your questions here:", "")
    if user_question:
        st.write(f"**Your question:** {user_question}")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        if docs:
            context = "\n".join([doc.page_content for doc in docs])
            chain = get_conversational_chain()
            response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write(f"**Answer:** {response['output_text']}")
        else:
            st.warning("⚠️ No relevant context found in the documents.")

def main():
    st.set_page_config(page_title="📄 Chat with PDF using Generative AI", page_icon="🤖", layout="wide")

    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin: 1rem;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 12px;
            font-size: 16px;
            padding: 10px 24px;
        }
        .stTextInput>div>div>input {
            background-color: #e8f0fe;
            color: #333;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        }
        .stFileUploader>div>label {
            color: #4CAF50;
            font-size: 16px;
            font-weight: bold;
        }
        .css-1d391kg p {
            font-size: 1.2rem;
            color: #333;
        }
        .stMarkdown {
            padding: 1rem;
            border-radius: 10px;
            background-color: #f7f7f7;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("📄 Chat with PDF using Generative AI 🤖")
    st.markdown(
        """
        Welcome to the **Chat with PDF** app! 🌟
        - Upload your PDF files 📄
        - Ask any questions you have 💬
        - Get detailed answers based on the document content 🤓
        """
    )

    uploaded_files = st.file_uploader("📂 Choose your PDF files", accept_multiple_files=True, type=['pdf'])

    if uploaded_files:
        st.markdown("### Files Uploaded Successfully! 🎉")
        text = get_text_from_pdf(uploaded_files)
        if text:
            chunks = get_chunks_from_text(text)
            get_vector_embeddings_from_chunks(chunks)
            st.success("PDF processed successfully! ✅")
            get_user_input()
        else:
            st.error("❌ No text extracted from the uploaded PDF files.")
    else:
        st.info("Please upload PDF files to get started. ⬆️")

if __name__ == "__main__":
    main()
