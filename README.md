
# 📄🤖 GenAI PDF Insight Bot

GenAI PDF Insight Bot is an innovative Streamlit application that leverages generative AI to interact with PDF documents. Upload PDFs and ask questions to receive detailed, context-aware answers based on the document content. Perfect for extracting knowledge from large PDFs and engaging with their content in a conversational manner.

## ✨ Features

- 📂 **Upload Multiple PDF Files**: Easily upload and manage multiple PDF documents.
- 🔍 **Extract and Process Text**: Automatically extract and process text from PDFs.
- 💬 **Ask Questions**: Interact with your documents by asking questions and receiving detailed answers.
- 🌈 **User-Friendly Interface**: Enjoy a colorful and intuitive interface for seamless interaction.

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Streamlit
- PyPDF2
- LangChain
- Google Generative AI
- FAISS
- Dotenv

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/genai-pdf-insight-bot.git
   cd genai-pdf-insight-bot
   ```

2. **Create a virtual environment and activate it**:
   ```bash
   python -m venv myenv
   myenv\Scripts\activate  # On Windows
   source myenv/bin/activate  # On macOS/Linux
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key**:
   - Create a `.env` file in the project root.
   - Add your Google Generative AI API key to the `.env` file:
     ```env
     GOOGLE_API_KEY=your_api_key_here
     ```

### Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Upload PDF Files**: Use the file uploader to select and upload your PDF files.
3. **Ask Questions**: Enter your questions in the text input box and receive detailed answers.



## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [PyPDF2](https://pythonhosted.org/PyPDF2/)
- [LangChain](https://langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Google Generative AI](https://cloud.google.com/ai-platform)
