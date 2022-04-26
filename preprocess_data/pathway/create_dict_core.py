import re, random
from pathlib import Path
from typing import List
import pickle
import json
from pprint import pprint
import os, sys
sys.path.append(os.path.abspath('.'))
from preprocess_data.pathway.util import get_logger, read_txt_file
import logging

root_dir = 'preprocess_data/pathway/'
data_dir = Path('data/Pathway/')
name_and_alias_in_parenthesis_pat = re.compile(r'^(.+)\((.+)\)$')
logger = get_logger(name=__name__, log_file=f'{root_dir}log.log', log_level=logging.DEBUG, log_level_name='')


def create_dict_core_only_one_type(name_file, entity_type, out_dir=None):
    """ create dict_core from entity names, only one type 
    Filter disease names and disease stop words

    1. TODO pattern, variant
    e.g. Loss of Function of SMAD2/3 in Cancer

    2. TODO spelling variants

    3 TODO. "pathway"(not "pathways") words and the continuously ahead words when the ahead word in dict_core.txt.
    except:
        2.0. the ahead word is a stop word.
        2.1. the ahead word not in the dict_core words.
        2.2. the ahead word 

    e.g. positive
    a I None S
    lysosomal I None S
    pathway I None S

    e.g. negtive, the ahead word not in the dict_core words.
    the I None S
    degradative I None S
    pathway I None S

    e.g. negtive, positive 
    selective I None S
    protein I None S
    import O None D
    pathway O None D

    e.g. positive, splits 'ubiquitin-proteasome' and search
    the I None S
    cytoplasmic I None S
    ubiquitin-proteasome I None S
    pathway I None S
    line 2585    
    """
    excluded_names = read_disease_stops_words()
    excluded_names += read_disease_words()

    dict_lines = []
    valiad_names = []
    with open(name_file, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip().lower()
            if not is_pathway_name_excluded(name, excluded_names):
                dict_lines.append(f'{entity_type}\t{line}')
                valiad_names.append(line)
    
    if not out_dir:
        out_dir = Path('data', entity_type)
    out_dict_file = Path(out_dir, 'dict_core.txt')
    out_name_file = Path(out_dir, f'{entity_type.lower()}_names.txt')

    with open(out_dict_file, 'w', encoding='utf-8') as f:
        for line in dict_lines:
            f.write(line)

    with open(out_name_file, 'w', encoding='utf-8') as f:
        for line in valiad_names:
            f.write(line)


def is_pathway_name_excluded(name, excluded_names):
    """ already pre lower case 
    e.g. 1
    Chagas disease (American trypanosomiasis)
    """
    if name in excluded_names:
        return True
    searched = name_and_alias_in_parenthesis_pat.search(name)
    if searched:
        if searched.group(1).strip() in excluded_names and searched.group(2).strip() in excluded_names:
            return True
    return False


def check_dictionary(names_file):
    pathway_names = read_txt_file(names_file)
    for i, item in enumerate(pathway_names):
        words = item.split()
        if len(words) == 1:
            logger.info(f'line {i} {words[0]}')


def read_disease_stops_words():
    disease_stops_words = []
    with open('data/stop_words/disease_stops_words.txt', 'r', encoding='utf-8') as f:
        for line in f:
            disease_stops_words.append(line.strip().lower())

    return disease_stops_words

def read_disease_words():
    disease_words = []
    with open('data/stop_words/disease_names.txt', 'r', encoding='utf-8') as f:
        for line in f:
            disease_words.append(line.strip().lower())
    return disease_words


if __name__ == "__main__":
    orig_name_file = 'data/Pathway/orig_pathway_names.txt'
    entity_type = 'Pathway'

    name_file = 'data/Pathway/pathway_names.txt'
    # check_dictionary(name_file)

    create_dict_core_only_one_type(orig_name_file, entity_type)
