# Speech and Language Processing

This repository contains the source files for the online book **"Speech and Language Processing"**. The book is built using [Jupyter Book](https://jupyterbook.org/).

---

## How to Build Locally

To build the book on your local machine, follow these steps:

### 1. Prerequisites
Ensure you have Python installed. It is recommended to use a virtual environment:

    python -m venv venv
    source venv/bin/activate

### 2. Install Jupyter Book
Install the latest version of Jupyter Book via pip:

    pip install jupyter-book==1.0.3

There may be advantages in using the newest version (2.X). But it doesn't work with my current jupyter book.

### 3. Build the Book
Navigate to the root directory (where _config.yml is located) and run:

    jb build .

### 4. View the Results
Open the generated HTML files in your browser:
- macOS: open _build/html/index.html
- Linux: xdg-open _build/html/index.html
- Windows: start _build/html/index.html

---

## Deployment to GitHub Pages

To host this book online, you can use the ghp-import tool:

1. Install ghp-import:
    pip install ghp-import

2. Push to GitHub Pages:
    ghp-import -n -p -f _build/html

Your book will be available at: https://chanwcom.github.io/speech_language_processing
