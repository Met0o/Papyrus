import os
from dotenv import load_dotenv
from imports import *

load_dotenv()

HF_AUTH=os.getenv('HF_AUTH')
MODEL_ID=os.getenv('MODEL_ID')
CONNECTION_STRING=os.getenv('CONNECTION_STRING')
COLLECTION_NAMES=["drugs", "papers", "medical"]
COLLECTION_PIPELINE_MAPPING = {"drugs": "DocumentsListRetriever",
                                "papers": "ContextualCompressionRetriever",
                                "medical": "ContextualCompressionRetriever"}