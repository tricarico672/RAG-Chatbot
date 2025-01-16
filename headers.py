import subprocess

def run_pip_installations():
    packages = [
        "langchain-unstructured",
        "unstructured-client",
        "chromadb",
        "langchain-chroma",
        "pytesseract",
        "backoff",
        "chromadb",
        "langgraph",
        "python-pptx",
        "unstructured[pptx]",
        "Cmake",
        "poppler-utils",
        "--upgrade setuptools wheel",
        "tesseract-ocr",
        "libreoffice",
        "pytesseract Pillow",
        "langchain_core",
        "bitsandbytes",
        "--upgrade langchain-huggingface",
        "unstructured-ingest",
        "cffi --only-binary :all:",
        "python-libmagic",
        "python-poppler",
        "--upgrade nltk",
        "unstructured[local-inference]"
        "streamlit"
        "boto3"
        "json"
        "langchain-aws"
    ]

    # Run pip commands to install each package
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run(f"!pip install {package}", shell=True, check=True)


