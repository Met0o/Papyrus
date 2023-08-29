from imports import *
from settings import *
from initialization import initialize_model, initialize_tokenizer, setup_components, initialize_generate_text, setup_prompt, setup_memory
from preprocessing import preprocess_document
from pipelines import execute_pipeline

app = Flask(__name__)

model = initialize_model()
tokenizer = initialize_tokenizer()
stores = setup_components()
prompt = setup_prompt()
memory = setup_memory()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_input = data["input"]
    collection = data.get("collection", COLLECTION_NAMES[0])
    store = stores[collection]

    generate_text = initialize_generate_text(model, tokenizer)
    llm = HuggingFacePipeline(pipeline=generate_text)

    retriever = MultiQueryRetriever.from_llm(llm=llm, 
                                            retriever=store.as_retriever(search_kwargs={"distance_metric": "cos",
                                                                                        "k":2}))

    res = execute_pipeline(llm, user_input, collection, retriever, prompt, memory)

    answer = res['result']

    docs = [doc.dict() for doc in res['source_documents']]
    for doc in docs:
        if "content" in doc:
            doc["content"] = preprocess_document(doc["content"])

    return jsonify({
        "answer": answer, 
        "source_documents": docs,
        "pipeline_used": COLLECTION_PIPELINE_MAPPING.get(collection, "unknown_pipeline")
    })

@app.route("/interact", methods=["POST"])
def interact():
    data = request.get_json()
    user_input = data["input"]

    generate_text = initialize_generate_text(model, tokenizer)
    llm = HuggingFacePipeline(pipeline=generate_text)

    response = llm(user_input)
    
    app.logger.info(f"LLM Response: {response}")
    
    try:
        answer = response
    except TypeError as e:
        app.logger.error(f"Error processing LLM response: {e}")
        return jsonify({"error": "Internal server error processing model response."}), 500
    
    return jsonify({
        "answer": answer
    })

@app.route('/collections', methods=['GET'])
def collections():
    return jsonify(COLLECTION_NAMES)

@app.route('/status', methods=['GET'])
def status():
    gpu_model = torch.cuda.get_device_name() if torch.cuda.is_available() else "Not using CUDA"
    return jsonify({"status": "Model loaded", "device": str(model.device), "gpu_model": gpu_model})

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({"error": "Internal server error. Please try again later."}), 500