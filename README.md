
# Papyrus

This is a Flask-based application designed to process and retrieve relevant information from documents using transformer models. The application can also be used for direct model interaction with selection from the controls.

This code is intended to run within Docker container with available GPU. By default, the application is configured to download and deploy the fastest (as of August 2023) 13B open-source model - Stable-Platypus2-13B. Thanks to Bitsandbytes, 4bit quantization makes the model fit into under 14GB VRAM. Llama-2-7B can be
 
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

- Python 3.10
- Required Libraries: Flask, transformers, torch, accelerate, bitsandbytes (compiled from source), langchain, psycopg2-binary, python-dotenv
- Identical version of CUDA (11.8) for both Torch and the docker image.

### Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/Met0o/local-llama2-gpu
   ```
2. Install Python packages:
   ```bash
   pip install Flask transformers torch accelerate bitsandbytes langchain psycopg2-binary python-dotenv
   ```

## Usage

1. Navigate to the project directory:
   ```bash
   cd path-to-your-directory/documentqna/app/refactored
   ```

2. Run the application:
   ```bash
   python main.py
   ```

3. The application will be available at `http://localhost:5000/`.

## Endpoints

- `/predict`: POST endpoint for prediction based on input data.
- `/collections`: GET endpoint to retrieve available collections.
- `/status`: GET endpoint to check the status of the model and device.

## Contributing

Any contributions are welcome.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

metodi.simeonov@gmail.com

Project Link: [https://github.com/Met0o/local-llama2-gpu](https://github.com/Met0o/local-llama2-gpu)