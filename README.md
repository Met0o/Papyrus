
# Papyrus

Papyrus is a Flask-based app crafted to retrieve information from vectorized documents in a `pgvector` database using `langchain` and transformer models. Moreover, its user interface, built on `Streamlit`, allows users to interact directly with the model without relying on the vector store.

To ensure the best performance, the code is designed to run in a Docker container with GPU support. By default, Papyrus will download and use the fastest open-source model as of August 2023: `Stable-Platypus2-13B`. Thanks to `Bitsandbytes` and 4-bit quantization, this model can operate using under 16GB of VRAM. As lightweight alternative, the `Llama-2-7b-chat-hf` model requires between 8-10GB of VRAM.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Endpoints](#endpoints)

## Getting Started

### Prerequisites

- Win11 or Linux OS with Docker and Docker Compose installed.
- NVIDIA GPU with at least 8GB of VRAM and 48GB of system RAM.

### Installation

Deployment configuration can be customized from the `docker-compose.yml` and individual `Dockerfiles`. `CUDA` version is set to 11.8 for `Torch`, Docker image, and `Bitsandbytes` which is compiled at first container runtime. The most important component of the deployment is the server image `nvidia/cuda:11.8.0-devel-ubuntu22.04`, which contains all necessary dependencies for complete GPU inference.

1. Clone the repository:
   ```bash
   git clone https://github.com/Met0o/Papyrus

## Usage

Use the .env file to set environment variables as needed.

You can build your own document RAG pipeline with a private vector store using the provisioned resources. With the help of `DocumentParser.py`, you can vectorize and embed your own PDF, docx, and csv files. 

The `thenlper/gte-large` embedding model, which utilizes a 1024-vector length, is incorporated into the configuration due to its optimal compatibility with `pgvector`. Moreover, using `Postgres` alongside the `pgvector` extension supports a hybrid operational model, combining both flat (relational) file structures and vectorized data.

1. Navigate to the project directory:
   ```bash
   cd path-to-your-directory/Papyrus
   ```

2. Build the Docker images:
   ```bash
   docker-compose build
   ```

3. Start the application using Docker Compose:
   ```bash
   docker-compose up
   ```

4. The application will be available at http://localhost:5000/ and the UI at http://localhost:8501/.

5.  Once the containers are active, connect to `Postgres` through the terminal, Docker, or your preferred database client to enable the `pgvector` extension:

```bash
CREATE EXTENSION vector;
```

## Endpoints

- `/predict`: A POST endpoint for inference based on embedded documents.
- `/interact`: A POST endpoint for direct model interactions without the need for embedded documents.
- `/collections`: A GET endpoint to list the available collections.
- `/status`: A GET endpoint to verify the model's status and the device it's running on.