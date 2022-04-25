import re, random
from pathlib import Path
from typing import List
import pickle
import json
from pprint import pprint
import os, sys


def create_dict_full(input_dir):
    """ create dict_full from entity names and high_quality_phrases
    only one type """
    full_phrases = set()
    exclued_filename_prefix = ('dict', 'raw', 'truth')
    
    orig_total_count = 0
    for file in Path(input_dir).glob('*.txt'):
        if not file.stem.startswith(exclued_filename_prefix):
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    full_phrases.add(line)
                    orig_total_count += 1
    
    print(f'orig_total_count {orig_total_count}, merged count {len(full_phrases)}')
    out_file = Path(input_dir, 'dict_full.txt')
    with open(out_file, 'w', encoding='utf-8') as f:
        for line in full_phrases:
            f.write(line)

input_dir = 'data/Pathway'
create_dict_full(input_dir)
