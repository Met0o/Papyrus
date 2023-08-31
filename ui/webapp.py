import requests
import streamlit as st
import time
import os

MAX_RETRIES = 60
RETRY_DELAY = 1

st.title("Papyrus")
st.sidebar.title("Controls")

if not st.session_state.get('model_loaded', False):
    
    info_message = st.sidebar.info("Please wait for model shards to be loaded into GPU memory...")

    for _ in range(MAX_RETRIES):
        try:
            response = requests.get("http://server:5000/status")
            if response.status_code == 200:
                info_message.success("Model is ready!")
                st.session_state.model_loaded = True
                break
        except requests.exceptions.ConnectionError:
            time.sleep(RETRY_DELAY)
    else:
        st.error("Failed to establish connection to the server, please check the app log.")

response = requests.get("http://server:5000/status")
status = response.json()
st.sidebar.markdown(f"{status['status']} on {status['device']}")
st.sidebar.markdown(f"**GPU Model:** {status['gpu_model']}")

mode = st.sidebar.radio("Application Mode", ["Document Retrieval", "Direct Model Interaction"])

response = requests.get("http://server:5000/collections")
collections = response.json()
selected_collection = st.sidebar.selectbox("Select a collection", collections)

def clear_chat_history():
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "How may I assist you today?"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and mode == "Document Retrieval":
            st.markdown("**Sources:**")
            for i, source in enumerate(message["sources"], start=1):
                st.markdown(f"{i}. {source}")

if prompt := st.chat_input("Ask your question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        
        start_time = time.time()
        try:
            if mode == "Document Retrieval":
                response = requests.post("http://server:5000/predict", 
                                        json={"input": prompt, 
                                            "collection": selected_collection})
            else:
                response = requests.post("http://server:5000/interact", 
                                        json={"input": prompt})

            response_data = response.json()
            answer = response_data.get("answer", "Please try again.")

            if mode == "Document Retrieval":
                pipeline_used = response_data.get("pipeline_used", "Unknown Pipeline")
                source_documents = response_data["source_documents"]
            else:
                pipeline_used = "Direct Interaction"
                source_documents = []

        except requests.exceptions.RequestException as e:
            answer = f"Error: Unable to connect to the server. {str(e)}"
            pipeline_used = "Unknown Pipeline"
            source_documents = []

        except requests.exceptions.JSONDecodeError:
            answer = "Error: Received an invalid response from the server."
            pipeline_used = "Unknown Pipeline"
            source_documents = []
        
        elapsed_time = time.time() - start_time
        st.sidebar.markdown(f"**Response time**: {round(elapsed_time, 2)} sec")
        st.sidebar.markdown(f"**Pipeline used:** {pipeline_used}")

        st.markdown(answer)
        
        if mode == "Document Retrieval":
            sources = list({os.path.basename(doc['metadata']['source']) for doc in source_documents})
            st.markdown("**Sources:**")
            for i, source in enumerate(sources, start=1):
                st.markdown(f"{i}. {source}")
            st.session_state.messages.append({"role": "assistant", 
                                            "content": answer, 
                                            "sources": sources})
        else:
            st.session_state.messages.append({"role": "assistant", 
                                              "content": answer})