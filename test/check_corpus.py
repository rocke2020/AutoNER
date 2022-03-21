import os, sys
sys.path.append(os.path.abspath('.'))
import re, random
from pathlib import Path
from typing import List
import pickle
import json
from pprint import pprint
import os, sys


annotaions_file = 'models/BC5CDR/annotations.ck'


def check_annotaions_file():
    with open(annotaions_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            items = line.split()
            if len(items) == 4:
                types = items[2].split(',')
                if len(types) > 1:
                    print(f'line num {i}, {types}')

check_annotaions_file()                