import streamlit as st
import os
from dotenv import load_dotenv
import uuid #generate unique id for every chunk
from pathlib import Path #tell the path to store ectorDB

from rag import process_pdf, generate_answer

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="Chat with PDF, RAG",
    layout="wide"
)
st.title("Chat with PDF using RAG")

#Create a new chat

def create_chat():
    chat_id= str(uuid.uuid4()) #uuid4 is used to create the unique ids for all the chunks

    st.session_state.chats[chat_id] = {
        "pdf_name": None,
        "summary": None,
        "vector_store": None,
        "messages": [],
        "language": "English"
    }

    st.session_state.active_chat = chat_id

# Initialize chat history
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "active_chat" not in st.session_state:
    create_chat()
chat = st.session_state.chats[st.session_state.active_chat]

with st.sidebar:

    if st.button("New Chat", use_container_width=True):
        create_chat()
        st.rerun()
    st.markdown("---")

    st.header("Chat History")
    for chat_id, chat_history in st.session_state.chats.items():
        if chat_history["pdf_name"]:
            title = chat_history["pdf_name"]
        else:
            title = "New Chat"
        if st.button(
            f"{title}",
            key=f"history_{chat_id}",
            use_container_width=True
        ):
            st.session_state.active_chat = chat_id
            st.rerun()

    # Language selection
    language = st.selectbox(
        "Choose your language",
        ["English", "Telugu", "Hindi"],
        key=f"language_{st.session_state.active_chat}",
        index=["English", "Telugu", "Hindi"].index(chat["language"])
    )


# PDF upload
if chat["summary"] is None:

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        key=f"upload_{st.session_state.active_chat}"
    )

    if uploaded_file and st.button("Process PDF", type="primary"):
        try:
            folder = Path("uploads")
            folder.mkdir(parents=True, exist_ok=True)

            pdf_path = folder / f"{st.session_state.active_chat}.pdf"
            pdf_path.write_bytes(uploaded_file.getvalue())

            chat["pdf_name"] = uploaded_file.name
            chat["language"] = language

            status = st.empty()
            result = None

            for step in process_pdf(pdf_path, language):
                if isinstance(step, str):
                    status.info(step)
                else:
                    result = step

            if result is None:
                raise ValueError("PDF processing failed.")
            chat["summary"] = result["summary"]
            chat["vector_store"] = result["vector_store"]

            status.success(result["status"])
            st.rerun()

        except Exception as e:
            st.error(f"Error processing PDF: {e}")


# Display the selected chat
if chat["summary"]:

    st.subheader(f"{chat['pdf_name']}")

    with st.expander("PDF Explanation", expanded=True):
        st.markdown(chat["summary"])

    st.divider()
    st.subheader("Ask Questions")

    # Display conversation
    for message in chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("View source pages"):
                    for source in message["sources"]:
                        st.markdown(f"Page {source['page']}")
                        st.write(source["content"])

    # Ask a question
    question = st.chat_input("Ask something about your PDF...")

    if question:
        chat["messages"].append({
            "role": "user",
            "content": question
        })

        try:
            answer, sources = generate_answer(
                question,
                chat["language"],
                chat["vector_store"]
            )

            chat["messages"].append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as e:
            chat["messages"].append({
                "role": "assistant",
                "content": f"Sorry, an error occurred: {e}"
            })

        st.rerun()

else:
    st.info("Upload a PDF to start this chat.")
