from imports import *

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