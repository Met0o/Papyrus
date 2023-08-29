from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores.pgvector import PGVector
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer
from langchain import PromptTemplate
from torch import cuda, bfloat16
from flask import Flask, request
from flask import make_response
from flask import jsonify
import transformers
import warnings
import logging
import torch
import os
import re

def preprocess_document(content):
    
    content = content.replace("\x0c", " ")
    content = re.sub(r'^\d+$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^Figure \d+\.\d+$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^Table \d+\.\d+$', '', content, flags=re.MULTILINE)
    headers = ["LIST OF TABLES", "List of Figures"]
    for header in headers:
        content = content.replace(header, "")
    content = re.sub(r'-\s*\n\s*Q', '', content)
    content = re.sub(r'\s+', ' ', content)
    content = content.strip()
    
    return content

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

torch.cuda.empty_cache()
warnings.filterwarnings('ignore', category=FutureWarning)

os.environ["LD_LIBRARY_PATH"] = "/usr/lib/cuda-11.8/lib64"

#model_id = 'meta-llama/Llama-2-7b-chat-hf'
model_id = 'garage-bAInd/Platypus2-7B'
hf_auth = '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'

CONNECTION_STRING = "postgresql+psycopg2://pgadmin:pgadmin@pgvector:5432/embeddings"
COLLECTION_NAMES = ["drugs", "papers", "medical"]
COLLECTION_PIPELINE_MAPPING = {
    "drugs": "DocumentsListRetriever",
    "papers": "ContextualCompressionRetriever",
    "medical": "ContextualCompressionRetriever"}

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16)

model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth)

model.eval()
print(f"**Model loaded on** {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,
    task='text-generation',
    #temperature=0.0,
    do_sample=False,
    max_new_tokens=1024,
    repetition_penalty=1.1)

llm = HuggingFacePipeline(pipeline=generate_text)
embeddings = HuggingFaceInstructEmbeddings(model_name='thenlper/gte-large')

stores = {collection: PGVector(
    collection_name=collection,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,) 
          for collection in COLLECTION_NAMES}

template = rf"""
CONTEXT:
-----------------------------------------------------------------------------
{'{context}'}
-----------------------------------------------------------------------------
PREVIOUS MESSAGES:
-----------------------------------------------------------------------------
{'{chat_history}'}
-----------------------------------------------------------------------------
QUESTION:
-----------------------------------------------------------------------------
{'{question}'}
-----------------------------------------------------------------------------

Your responses should be complete sentences, in clear and concise tone.

Answer:"""

prompt = PromptTemplate(
    input_variables=["chat_history","question", "context"], 
    template=template)

memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            input_key="question",
            k=3,
            return_messages=True)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        user_input = data["input"]
        collection = data.get("collection", COLLECTION_NAMES[0])
        store = stores[collection]
        
        retriever = MultiQueryRetriever.from_llm(llm=llm, 
                                                retriever=store.as_retriever(search_kwargs={"distance_metric": "cos",
                                                                                            "k":2}))
        
        res = {"result": "Error: Unexpected pipeline selected.", "source_documents": []}
        
        selected_pipeline_type = COLLECTION_PIPELINE_MAPPING.get(collection, "unknown_pipeline")
        
        if selected_pipeline_type == "DocumentsListRetriever":
            rag_pipeline = RetrievalQA.from_chain_type(llm=llm, 
                                                    chain_type="stuff", 
                                                    retriever=retriever,
                                                    return_source_documents=True, 
                                                    verbose=True,
                                                    chain_type_kwargs={
                                                        "verbose": True,
                                                        "prompt": prompt,
                                                        "memory": memory})
            res = rag_pipeline(user_input)
        
        elif selected_pipeline_type == "ContextualCompressionRetriever":
            compressor = LLMChainExtractor.from_llm(llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=retriever)
            
            rag_pipeline = RetrievalQA.from_chain_type(llm=llm, 
                                        retriever=compression_retriever,
                                        return_source_documents=True, 
                                        verbose=True,
                                        chain_type_kwargs={
                                            "verbose": True,
                                            "prompt": prompt,
                                            "memory": memory})
            res = rag_pipeline(user_input)

        else:
            logging.error(f"Unexpected pipeline type: {selected_pipeline_type}")

        answer = res['result']
        
        docs = [doc.dict() for doc in res['source_documents']]
        for doc in docs:
            if "content" in doc:
                doc["content"] = preprocess_document(doc["content"])

        return jsonify({
            "answer": answer, 
            "source_documents": docs,
            "pipeline_used": selected_pipeline_type
        })
    except Exception as e:
        logging.error(f"Error in predict route: {str(e)}")
    return make_response(jsonify(error=str(e)), 500)

@app.route('/collections', methods=['GET'])
def collections():
    return jsonify(COLLECTION_NAMES)

@app.route('/status', methods=['GET'])
def status():
    gpu_model = torch.cuda.get_device_name() if torch.cuda.is_available() else "Not using CUDA"
    return jsonify({"status": "Model loaded", "device": str(model.device), "gpu_model": gpu_model})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)