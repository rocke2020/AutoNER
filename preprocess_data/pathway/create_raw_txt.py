import re, random
from pathlib import Path
from typing import List
import pickle
import json
from pprint import pprint
import os, sys
from nltk import TreebankWordTokenizer
from nltk.tokenize import word_tokenize

root_dir = Path('/home/qcdong/corpus/NER/autoner/pathway')
json_file = Path('/home/qcdong/corpus/NER/autoner/pathway', 'pathway_pubmed_sentences.json')
text_file = Path('/home/qcdong/corpus/NER/autoner/pathway', 'pathway_pubmed_sentences.txt')


def get_raw_txt():
    # tokenizer = TreebankWordTokenizer()
    out_dir = Path('data/Pathway')
    out_file = out_dir / 'raw_text.txt'
    with open(out_file, 'w', encoding='utf-8') as f_out:
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                words = word_tokenize(line.strip())
                for word in words:
                    f_out.write(f'{word}\n')
                f_out.write('\n')

get_raw_txt()