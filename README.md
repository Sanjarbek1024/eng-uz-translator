# EN → UZ Translator

A neural machine translation app — English to Uzbek — built with a Seq2Seq GRU model with Bahdanau attention.

## Model Architecture

- **Encoder**: Bidirectional GRU, 256 hidden units, shared SentencePiece (BPE) vocabulary (16k tokens)
- **Decoder**: Unidirectional GRU with Bahdanau-style attention
- **Tokenizer**: SentencePiece BPE trained jointly on English + Uzbek

## Project Structure

```
en-uz-translator/
├── app/
│   ├── __init__.py
│   ├── model.py        # Encoder, Decoder, Seq2Seq classes
│   └── translator.py   # load_model() and translate() functions
├── saved_model/
│   ├── full_seq2seq_checkpoint.pth   # trained weights (not in git)
│   └── spm.model                     # SentencePiece tokenizer (not in git)
├── static/             # CSS / JS assets (optional)
├── templates/
│   └── index.html      # single-page frontend
├── main.py             # FastAPI app
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. clone and enter
git clone https://github.com/Sanjarbek1024/eng-uz-translator.git
cd en-uz-translator

# 2. create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4. add your model files
#    copy full_seq2seq_checkpoint.pth and spm.model into saved_model/

# 5. run
uvicorn main:app --reload
```

Open http://localhost:8000

## Deploying to GitHub

```bash
cd en-uz-translator
git init
git add .
git commit -m "initial commit"

# create a repo on github.com, then:
git remote add origin https://github.com/Sanjarbek1024/eng-uz-translator.git
git branch -M main
git push -u origin main
```

> The `saved_model/` folder is in `.gitignore` because model files are large.
> Upload them separately (Google Drive, Hugging Face Hub, etc.) and document the link in your README.
