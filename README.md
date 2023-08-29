
# Papyrus

Papyrus is an app built on Flask that is designed to extract and fetch information from vectorized documents stored in a `pgvector` database. It does this by utilizing `langchain` and transformer models. Also, the application has a feature that enable users to interact directly with the model through an UI based on `Streamlit`.

To ensure the best performance, the code is designed to run in a Docker container with GPU support. By default, Papyrus will download and use the fastest open-source model as of August 2023: `Stable-Platypus2-13B`. Thanks to `Bitsandbytes` and their 4-bit quantization, this model can operate using just under 14GB of VRAM. As an alternative, the `Llama-2-7b-chat-hf` model is compatible and requires between 8-10GB of VRAM.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Endpoints](#endpoints)

## Getting Started

### Prerequisites

- Docker and Docker Compose installed.
- An environment with NVIDIA GPU.

### Installation

Deployment configuration can be customized from the `docker-compose.yml` and individual `Dockerfiles`. `CUDA` version is set to 11.8 for `Torch`, Docker image, and `Bitsandbytes` which is compiled at first container runtime.

1. Clone the repository:
   ```bash
   git clone https://github.com/Met0o/Papyrus

## Usage

Use the .env file to set environment variables as needed.
Docker containers can be created under Windows and Linux, so there are no OS restrictions.

You can build your own document RAG pipeline with a private vector store using the provisioned resources. With the help of `DocumentParser.py`, you can vectorize and embed your own PDF, docx, and csv files. 

The embedding model included in the configuration is the `thenlper/gte-large` which proved to be the best fit for the `pgvector` as it uses 1024 vector length. Additionally, `Postgres` with `pgvector` extension enables for a hybrid model of operation - flat file structure and vectorized data.

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

5. Once containers are up and running, connect to the postgres from terminal/docker or your db client of choice and run: 

```bash
CREATE EXTENSION vector;
```

## Endpoints

- `/predict`: A POST endpoint for inference based on embedded documents.
- `/interact`: A POST endpoint for direct model interactions without the need for embedded documents.
- `/collections`: A GET endpoint to list the available collections.
- `/status`: A GET endpoint to verify the model's status and the device it's running on.