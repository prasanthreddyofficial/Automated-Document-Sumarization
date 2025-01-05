import os
from flask import Flask, render_template, request
from transformers import T5Tokenizer, T5ForConditionalGeneration
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from docx import Document

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf", "txt", "docx"}

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(filepath):
    _, ext = os.path.splitext(filepath)
    if ext == ".pdf":
        reader = PdfReader(filepath)
        return " ".join(page.extract_text() for page in reader.pages)
    elif ext == ".docx":
        doc = Document(filepath)
        return " ".join(para.text for para in doc.paragraphs)
    elif ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    return ""

@app.route("/", methods=["GET", "POST"])
def home():
    summary = ""
    document = ""
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file part.")
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No selected file.")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            document = extract_text_from_file(filepath)
            input_text = "summarize: " + document
            inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
            summary_ids = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            os.remove(filepath)
        else:
            return render_template("index.html", error="Invalid file type.")
    return render_template("index.html", document=document, summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
