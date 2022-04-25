import re, random
from pathlib import Path
from typing import List
import pickle
import json
from pprint import pprint
import os, sys
import os, sys
sys.path.append(os.path.abspath('.'))
from preprocess_data.pathway.util import get_logger, read_txt_file
import logging


root_dir = 'preprocess_data/pathway/'
data_dir = Path('data/Pathway/')
logger = get_logger(name=__name__, log_file=f'{root_dir}log.log', log_level=logging.DEBUG, log_level_name='')


"""  Only 5 filter rules, positive, lower case 
1. filter only too comman disease stops words.
e.g. positive
and I None S
disease I Pathway S
. I None S

2. "pathway"(not "pathways") words and the continuously ahead words when the ahead word in dict_core.txt.
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


pathway_names_file = data_dir / 'pathway_names.txt'




def check_dictionary(names_file):
    pathway_names = read_txt_file(names_file)
    for item in pathway_names:
        words = item.split()
        if len(words) == 1:
            logger.info(words[0])

check_dictionary(pathway_names_file)
