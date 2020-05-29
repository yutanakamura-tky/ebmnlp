import nltk
from .ebmnlp_bioelmo_crf import EBMNLPTagger
from flask import Flask, request, render_template

app = Flask(__name__)

EBMNLP_BIOELMO_CRF_CHECKPOINT_PATH = './models/ebmnlp_bioelmo_crf/ebmnlp_bioelmo_crf.ckpt'

@app.route('/')
def form():
    return render_template('sample.html')

@app.route('/predict', methods=['POST'])
def predict():
    abstract = request.form['abstract']
    use_cuda = request.form['use_cuda']
    ebmnlp = EBMNLPTagger.load_from_checkpoint(EBMNLP_BIOELMO_CRF_CHECKPOINT_PATH)

    if bool(use_cuda):
        ebmnlp.to('cuda')

    tokens = nltk.word_tokenize(abstract)
    tags = ebmnlp.unpack_pred_tags(ebmnlp.forward([tokens]))
    tagging = [(tag, token) for tag, token in zip(tags[0], tokens)]
    return render_template('sample.html', tagging=tagging)
