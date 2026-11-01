
#operations performed in this file:
# 1. exporting data from PDFs
# 2. Chunking data
# 3. storing chunks into FAISS
# 4. Sending and Receiving Query/results from LLM

#Library required
from uuid import uuid4 #generate unique id for every chunk
from dotenv import load_dotenv #read the .env file for API key
from pathlib import Path #tell the path to store ectorDB
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings



#load_dotenv()

#Initialize variables 
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" #embedding - convert chunks into vectors
embedding = None
llm = None
vector_store = None
def initialize_components():
    global llm, embedding

    if llm is None:
        llm = ChatGroq(
            model = "openai/gpt-oss-20b",
            temperature = 0.7, #Controls Hallucination
            max_tokens=1000 #LLm can generate one pass in one token.....(max length of answer given by LLM)
        )
    if embedding is None:
        embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
def process_pdf(pdf_path, language): #yield will show the message on the screen
    global vector_store

    yield "Initializing components..." #initializing llms, embedding models and vector
    initialize_components()

    yield "Loading data from PDF..."
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    if not documents:
        raise ValueError("No readable pages found in the PDF.")

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap= 200
    )

    docs = splitter.split_documents(documents)

    if not docs:
        raise ValueError("No readable text found in the PDF.")

    yield "Storing chunks in FAISS..."

    vector_store = FAISS.from_documents( docs, embedding )

    yield "Generating PDF explanation..."

    # Combine extracted PDF text for the initial explanation
    content = "\n\n".join( doc.page_content for doc in documents )
    content = content[:15000]

    prompt = f"""
You are a helpful PDF assistant.

Explain the main points of the PDF in {language}.
Use simple language and bullet points.
Include important details when available.
Do not invent information.

PDF content:
{content}
"""
    response = llm.invoke(prompt)
    summary = response.content.strip()

    # Return the explanation and the created vector store
    yield {
        "status": "PDF processed successfully!",
        "summary": summary,
        "chunks": len(docs),
        "vector_store": vector_store
    }

# Generate answers from the selected PDF
def generate_answer(query, language, selected_vector_store=None):
    # Use the vector store belonging to the selected chat
    if selected_vector_store is None:
        selected_vector_store = vector_store

    if selected_vector_store is None:
        raise RuntimeError(
            "Please upload and process a PDF first."
        )
    initialize_components()

    retriever = selected_vector_store.as_retriever()

    docs = retriever.invoke(query)

    content = "\n\n".join(
        doc.page_content for doc in docs
    )

    if not content.strip():
        return (
            "I could not find the answer in the uploaded PDF.",
            []
        )

    prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the question only using the content given below.
Respond in {language}.
If the answer is not present, say:
"I could not find the answer in the uploaded PDF."
Do not hallucinate or invent information.

Content:
{content}

Question:
{query}
"""
    response = llm.invoke(prompt)
    # Collect source page numbers
    sources = []

    for doc in docs:
        page_number = doc.metadata.get("page", 0) + 1

        sources.append({
            "page": page_number,
            "content": doc.page_content[:500]
        })

    return response.content.strip(), sources