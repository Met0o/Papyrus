from imports import *
from settings import *

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEVICE = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

def initialize_model():
    torch.cuda.empty_cache()
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16)
    
    model_config = transformers.AutoConfig.from_pretrained(
        MODEL_ID,
        use_auth_token=HF_AUTH)
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=HF_AUTH)
    
    model.eval()
    return model

def initialize_tokenizer():
    return transformers.AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_AUTH)

def initialize_generate_text(model, tokenizer):
    return transformers.pipeline(
        model=model, 
        tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        do_sample=False,
        max_new_tokens=1024,
        repetition_penalty=1.1)

def setup_components():
    embeddings = HuggingFaceInstructEmbeddings(model_name='thenlper/gte-large')

    stores = {collection: PGVector(
        collection_name=collection,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings) 
              for collection in COLLECTION_NAMES}
    
    return stores

def setup_prompt():
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

    return PromptTemplate(
        input_variables=["chat_history", "question", "context"], 
        template=template)

def setup_memory():
    return ConversationBufferWindowMemory(
            memory_key="chat_history",
            input_key="question",
            k=3,
            return_messages=True)