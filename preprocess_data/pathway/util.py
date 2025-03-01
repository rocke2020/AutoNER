import logging, os


_nameToLevel = {
#    'CRITICAL': logging.CRITICAL,
#    'FATAL': logging.FATAL,  # FATAL = CRITICAL
#    'ERROR': logging.ERROR,
#    'WARN': logging.WARNING,
   'INFO': logging.INFO,
   'DEBUG': logging.DEBUG,
}


def get_logger(name=__name__, log_file=None, log_level=logging.DEBUG, log_level_name=''):
    """ default log level DEBUG """
    logger = logging.getLogger(name)
    if name == 'app':
        fmt= '%(asctime)s %(filename)10s %(levelname)s L %(lineno)d: %(message)s'
    else:
        fmt= '%(asctime)s %(name)s %(levelname)s L %(lineno)d: %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(format=fmt, datefmt=datefmt)
    if log_file is not None:
        log_file_folder = os.path.split(log_file)[0]
        if log_file_folder:
            os.makedirs(log_file_folder, exist_ok=True)
        fh = logging.FileHandler(log_file, 'w', encoding='utf-8')
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(fh)
    if log_level_name in _nameToLevel:
        log_level = _nameToLevel[log_level_name]
    logger.setLevel(log_level)
    return logger


def read_txt_file(text_file):
    pathway_names = []
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            pathway_names.append(line.strip())
    return pathway_names   


def split_words(s:str):
    words = []
    raw_words = s.split()
    for raw_word in raw_words:
        pass

if __name__ == "__main__":
    ss = [

    ]
    for sentence in ss:
        print(split_words(sentence))