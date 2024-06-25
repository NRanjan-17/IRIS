

# Iris: Your Personal Health Companion

Iris is a personalized health companion bot that provides accurate medical information and advice using advanced language models, featuring conversation logging and easy-to-use default prompts for quick inquiries.

## Prerequisites

Before you can start using the IRIS Bot, make sure you have the following prerequisites installed on your system:

- Python 3.6 or higher
- Required Python packages (you can install them using pip):
    - langchain
    - chainlit
    - sentence-transformers
    - faiss
    - PyPDF2 (for PDF document loading)

## Installation

1. Clone this repository to your local machine.

    ```bash
    git clone https://github.com/NRanjan-17/IRIS.git
    cd IRIS
    ```

2. Create a Python virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```

3. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Download the required language models and data. Please refer to the Langchain documentation for specific instructions on how to download and set up the language model and vector store.

5. Set up the necessary paths and configurations in your project, including the `DB_FAISS_PATH` variable and other configurations as per your needs.

## How to start Bot

Use the following commang to start the bot

1. Provide the dat in the pdf file is data dir and run the `ingest.py` script to create the vector store.

2. After the vector store is created start the bot.

```bash
    python -m chainlit run model.py
    ```