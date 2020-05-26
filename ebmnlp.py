import argparse
import nltk
from ebmnlp_bioelmo_crf import EBMNLPTagger
nltk.download('punkt')

def predict(config):
    """
    NameSpace -> list(tuple)
    """
    ebmnlp = EBMNLPTagger.load_from_checkpoint('./lightning_logs/version_146/checkpoints/epoch=13.ckpt')

    if not config.no_cuda:
        ebmnlp.to('cuda')

    with open(config.filename) as f:
        tokens = nltk.word_tokenize(''.join(f.readlines()))

    tags = ebmnlp.unpack_pred_tags(ebmnlp.forward([tokens]))
    tagging = [(tag, token) for tag, token in zip(tags[0], tokens)]

    if config.out_filename:
        with open(config.out_filename, mode='w') as f:
            f.write('\n'.join([f'{tag}\t{token}' for tag, token in zip(tags[0], tokens)]))
        print(f'Tagging done! -> {config.out_filename}')

    else:
        print('\n'.join([f'{tg[0]}\t{tg[1]}' for tg in tagging]))

    return tagging

if __name__=='__main__':

    def get_args():
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument(dest='filename', help='path of txt file to read')
        parser.add_argument(dest='out_filename', nargs='?', help='path of txt file to read')
        parser.add_argument('--no-cuda', action='store_true', dest='no_cuda', help='set to use without CUDA')
        args = parser.parse_args()
        return args

    config = get_args()
    predict(config)
