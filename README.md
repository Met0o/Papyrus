
# Papyrus

Papyrus is an app built on Flask that is designed to extract and fetch information from vectorized documents stored in a pgvector database. It does this by utilizing langchain and transformer models. Also, the application has a feature that enable users to interact directly with the model through the Streamlit UI.

For optimal performance, this code is meant to operate inside a Docker container that supports GPU. Out of the box, Papyrus is set to download and use the latest and fastest open-source model as of August 2023: ```Stable-Platypus2-13B```. With the help of Bitsandbytes and 4-bit quantization it is possible for this model to run on just under 14GB of VRAM. Alternatively, ```Llama-2-7b-chat-hf``` is also compatible and can fit in 8-10GB of VRAM.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Endpoints](#endpoints)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Getting Started

### Prerequisites

- Docker and Docker Compose installed.
- An environment with NVIDIA GPU.

### Installation

Deployment configuration can be customized from the docker-compose.yml and individual Dockerfiles. CUDA version is set to 11.8 for Torch, Docker image, and Bitsandbytes which is compiled at first container runtime.

1. Clone the repository:
   ```bash
   git clone https://github.com/Met0o/Papyrus

## Usage

You can build your own document RAG pipeline with a private vector store using the provision resources and with the help of ```DocumentParser.py```, you can vectorize and embed your PDF, docx, and csv files with ease. 

The embedding model included in the configuration is the ```thenlper/gte-large``` which proved to be the best fit for the pgvector as it uses 1024 vector length and enables for a hybrid model of operation.

Once containers are up and running, connect to the postgres from terminal/docker or your db client of choice and run: 

```bash
CREATE EXTENSION vector;
```

1. Navigate to the project directory:
   ```bash
   cd path-to-your-directory/Papyrus
   ```

2. Build the Docker images:
   ```bash
   docker-compose build
   ```

2. Start the application using Docker Compose:
   ```bash
   docker-compose up
   ```

3. The application will be available at http://localhost:5000/ and the UI at http://localhost:8501/.

## Usage

Use the .env file to set environment variables as needed.
Docker containers can be created under Windows and Linux, no OS restrictions.

## Endpoints

- `/predict`: POST endpoint for prediction based on embedded documents.
- `/interact`: POST endpoint for inference without using embedded documents.
- `/collections`: GET endpoint to retrieve available collections.
- `/status`: GET endpoint to check the status of the model and device.

## Contact

metodi.simeonov@gmail.com

Project Link: [https://github.com/Met0o/Papyrus](https://github.com/Met0o/Papyrus)