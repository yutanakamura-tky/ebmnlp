import argparse
import nltk
from ebmnlp_bioelmo_crf import EBMNLPTagger

nltk.download('punkt')

EBMNLP_BIOELMO_CRF_CHECKPOINT_PATH='models/ebmnlp_bioelmo_crf/ebmnlp_bioelmo_crf.ckpt'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str, nargs='?') 
    parser.add_argument('--no-cuda', dest='no_cuda', action='store_true') 

    args = parser.parse_args()
    return args


def main():
    config = get_args()

    with open(config.input_file) as f:
        abstract = ''.join(f.readlines())

    ebmnlp = EBMNLPTagger.load_from_checkpoint(EBMNLP_BIOELMO_CRF_CHECKPOINT_PATH)

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
        print(f'Tagging done! -> {config.out_filename}')


if __name__=='__main__':
    main()
