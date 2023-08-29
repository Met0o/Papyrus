from settings import COLLECTION_PIPELINE_MAPPING
from imports import *

def execute_pipeline(llm, user_input, collection, retriever, prompt, memory):
    
    selected_pipeline_type = COLLECTION_PIPELINE_MAPPING.get(collection, "unknown_pipeline")
    
    res = {"result": "Error: Unexpected pipeline selected.", "source_documents": []}

    if selected_pipeline_type == "DocumentsListRetriever":
        rag_pipeline = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=True, 
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": prompt,
                "memory": memory}
        )
        res = rag_pipeline(user_input)

    elif selected_pipeline_type == "ContextualCompressionRetriever":
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=retriever
        )
        
        rag_pipeline = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=compression_retriever,
            return_source_documents=True, 
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": prompt,
                "memory": memory}
        )
        res = rag_pipeline(user_input)

    else:
        logging.error(f"Unexpected pipeline type: {selected_pipeline_type}")

    return res