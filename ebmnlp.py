import argparse
import nltk
from ebmnlp_bioelmo_crf import EBMNLPTagger

nltk.download('punkt')

def main(config):
    with open(config.input_file) as f:
        abstract = ''.join(f.readlines())

    ebmnlp = EBMNLPTagger.load_from_checkpoint('./checkpoint/ebmnlp_bioelmo_crf.ckpt')

    if not bool(config.no_cuda):
        ebmnlp.to('cuda')

    tokens = nltk.word_tokenize(abstract)
    tags = ebmnlp.unpack_pred_tags(ebmnlp.forward([tokens]))
    tagging = [(tag, token) for tag, token in zip(tags[0], tokens)]
    result_str = '\n'.join([f'{tg[0]}\t{tg[1]}' for tg in tagging])

    config.output_file = config.output_file or None

    if config.output_file is None:
        print(result_str)

    else:
        with open(config.output_file, 'w') as f:
            f.write(result_str)


if __name__=='__main__':
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('input_file', type=str)
        parser.add_argument('output_file', type=str, nargs='?') 
        parser.add_argument('--no-cuda', dest='no_cuda', action='store_true') 

    config = parser.parse_args()
    return config

    main(config)
