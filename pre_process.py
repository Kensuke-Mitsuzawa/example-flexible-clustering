#! -*- coding: utf-8 -*-
import os
import JapaneseTokenizer
import codecs
import copy
import jsonlines
from collections import Counter
import tqdm
import logging
logging.basicConfig(level=logging.INFO)

"""ライブドアコーパスの前処理を実行する
1. テキストファイルを読み込む
2. 形態素分割する
3. 単語の頻度カウントをする（SIFの計算に必要）
3. 形態素分割結果はjsonlに保存する
"""

MODEL_OBJ = {"file_name": "", "title_morphs": "", "category": "", "morphs": "", "timestamp": "", "url": ""}
mecab_obj = JapaneseTokenizer.MecabWrapper(dictType='ipadic')


def load_liverdoor_corpus(path_text_file: str, file_name: str, category: str):
    f_obj = codecs.open(path_text_file, 'r')
    url = f_obj.readline().strip()
    timestamp = f_obj.readline().strip()
    document_title = f_obj.readline().strip()

    __morphs = []
    for t in f_obj.readlines():
        __morphs += mecab_obj.tokenize(t, is_surface=True, return_list=True)

    model_obj = copy.deepcopy(MODEL_OBJ)
    model_obj['title_morphs'] = mecab_obj.tokenize(document_title, is_surface=True, return_list=True)
    model_obj['file_name'] = file_name
    model_obj['category'] = category
    model_obj['morphs'] = __morphs
    model_obj['timestamp'] = timestamp
    model_obj['url'] = url

    return model_obj


def main(path_corpus_root: str,
         path_processed_jsonl: str,
         path_morph_freq_file: str):
    __morphs = []
    fp = open(path_processed_jsonl, 'w')
    writer = jsonlines.Writer(fp)
    for root, subdirs, files in os.walk(path_corpus_root):
        logging.info('processing {}...'.format(files))
        if len(subdirs) > 0:
            continue
        for f_name in tqdm.tqdm(files):
            __root, dir_name = os.path.split(root)
            path_text_file = os.path.join(root, f_name)
            processed_obj = load_liverdoor_corpus(path_text_file, f_name, dir_name)
            writer.write(processed_obj)
            __morphs += processed_obj['morphs']
        else:
            pass
    else:
        writer.close()
        fp.close()

    morphs_freq = [(k, v) for k, v in Counter(__morphs).items()]
    fp = open(path_morph_freq_file, 'w')
    for t in morphs_freq:
        fp.write('{} {}'.format(t[0], t[1]) + '\n')
    else:
        fp.close()


def __main():
    pass


def test():
    path_corpus_root = './text'
    path_processed_jsonl = './text/processed_data.jsonl'
    path_morph_freq_file = './text/vocab.txt'

    main(path_corpus_root, path_processed_jsonl, path_morph_freq_file)


if __name__ == '__main__':
    test()
