# Papyrus

Papyrus is a Flask-based application designed to process and retrieve relevant information from documents using transformer models. Additionally, the application can facilitate direct model interaction with selections from the controls.

This code is intended to run within a Docker container with GPU support. By default, the application is configured to download and deploy the fastest 13B open-source model as of August 2023 - Stable-Platypus2-13B. Thanks to Bitsandbytes, 4-bit quantization allows the model to fit into under 14GB VRAM. The Llama-2-7B can also be utilized.

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
- An environment with GPU support.
- CUDA version 11.8 compatible with Torch and the Docker image.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Met0o/local-llama2-gpu
Navigate to the project directory:

bash
Copy code
cd path-to-your-directory/documentqna/app/refactored
Build the Docker image:

bash
Copy code
docker-compose build
Start the application using Docker Compose:

bash
Copy code
docker-compose up
The application will be available at http://localhost:5000/ and the UI at http://localhost:8501/.

Usage
Use the .env file to set environment variables as needed.
Ensure the Docker containers are up and running using the docker-compose up command.
Access the web interface via your browser to interact with the model.
Endpoints
/predict: POST endpoint for prediction based on input data.
/collections: GET endpoint to retrieve available collections.
/status: GET endpoint to check the status of the model and device.
Contributing
Any contributions are welcome.

Fork the Project
Create your Feature Branch (git checkout -b feature/AmazingFeature)
Commit your Changes (git commit -m 'Add some AmazingFeature')
Push to the Branch (git push origin feature/AmazingFeature)
Open a Pull Request
Contact
metodi.simeonov@gmail.com

Project Link: https://github.com/Met0o/local-llama2-gpu
