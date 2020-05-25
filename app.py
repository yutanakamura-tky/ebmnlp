import nltk
from ebmnlp_bioelmo_crf import EBMNLPTagger
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def form():
    return render_template('sample.html')

@app.route('/predict', methods=['POST'])
def predict():
    abstract = request.form['abstract']
    ebmnlp = EBMNLPTagger.load_from_checkpoint('./lightning_logs/version_146/checkpoints/epoch=13.ckpt')
    ebmnlp.to('cuda')
    tokens = nltk.word_tokenize(abstract)
    tags = ebmnlp.unpack_pred_tags(ebmnlp.forward([tokens]))
    tagging = [(tag, token) for tag, token in zip(tags[0], tokens)]
    return render_template('sample.html', tagging=tagging)
