#!/bin/bash
MODEL_NAME="Pathway"
RAW_TEXT=data/$MODEL_NAME/raw_text.txt
DICT_CORE=data/$MODEL_NAME/dict_core.txt
DICT_FULL=data/$MODEL_NAME/dict_full.txt

green=`tput setaf 2`
reset=`tput sgr0`

MODEL_ROOT=./models/$MODEL_NAME
TRAINING_SET=$MODEL_ROOT/annotations.ck

mkdir -p $MODEL_ROOT

echo ${green}=== Compilation ===${reset}
make

echo ${green}=== Generating Distant Supervision ===${reset}
bin/generate $RAW_TEXT $DICT_CORE $DICT_FULL $TRAINING_SET

echo ${green}Done.${reset}
