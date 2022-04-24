import re, random
from pathlib import Path
from typing import List
import pickle
import json
from pprint import pprint
import os, sys




def create_dict_core_only_one_type(name_file, entity_type, out_dir=None):
    """ create dict_core from entity names, only one type """
    dict_lines = []
    with open(name_file, 'r', encoding='utf-8') as f:
        for line in f:
            dict_lines.append(f'{entity_type}\t{line}')
    
    if out_dir:
        out_file = Path(out_dir, 'dict_core.txt')
    else:
        out_file = Path('data', entity_type, 'dict_core.txt')

    with open(out_file, 'w', encoding='utf-8') as f:
        for line in dict_lines:
            f.write(line)

name_file = 'data/Pathway/pathway_names.txt'
entity_type = 'Pathway'
create_dict_core_only_one_type(name_file, entity_type)
