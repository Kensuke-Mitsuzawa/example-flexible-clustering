#! -*- coding: utf-8 -*-
import os
import JapaneseTokenizer
import codecs
import copy

"""ライブドアコーパスの前処理を実行する
1. テキストファイルを読み込む
2. 形態素分割する
3. 単語の頻度カウントをする（SIFの計算に必要）
3. 形態素分割結果はjsonlに保存する
"""

MODEL_OBJ = {"title": "", "category": "", "morphs": ""}
mecab_obj = JapaneseTokenizer.MecabWrapper(dictType='ipadic')


def load_liverdoor_corpus(path_text_file: str, file_name: str, category: str):
    f_obj = codecs.open(path_text_file, 'r')
    # はじめの２行はURLとタイムスタンプなので，飛ばす
    f_obj.readline()
    f_obj.readline()

    __morphs = []
    for t in f_obj.readlines():
        __morphs += mecab_obj.tokenize(t, is_surface=True, return_list=True)

    model_obj = copy.deepcopy(MODEL_OBJ)
    model_obj['title'] = file_name
    model_obj['category'] = category
    model_obj['morphs'] = __morphs

    return model_obj


def main(path_corpus_root: str):
    __stack = []
    for root, subdirs, files in os.walk(path_corpus_root):
        if len(subdirs) > 0:
            continue
        for f_name in files:
            __root, dir_name = os.path.split(root)
            path_text_file = os.path.join(root, f_name)
            processed_obj = load_liverdoor_corpus(path_text_file, f_name, dir_name)
            __stack.append(processed_obj)
        else:
            pass
    else:
        pass


def __main():
    pass


def test():
    path_corpus_root = './text'
    main(path_corpus_root)


if __name__ == '__main__':
    test()
