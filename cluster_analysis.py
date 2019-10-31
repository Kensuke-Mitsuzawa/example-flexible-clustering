#! -*- coding: utf-8 -*-
import csv
from typing import List, Dict, Any
import itertools
import logging
import statistics
import json
import jsonlines
import tqdm
from DocumentFeatureSelection import interface
logging.basicConfig(level=logging.INFO)


def load_leaf_table(path_cluster_leaf_table: str) -> List[Any]:
    f_obj = open(path_cluster_leaf_table, 'r')
    reader = csv.reader(f_obj, delimiter='\t')
    next(reader)
    __ = list(reader)
    f_obj.close()
    return __


def load_preprocessed_record(path_preprocessed_jsonl: str) -> Dict[str, List[str]]:
    with jsonlines.open(path_preprocessed_jsonl) as reader:
        __ = {record['file_name']: record['morphs'] for record in tqdm.tqdm(reader)}
    return __


# クラスタリング結果のテーブル
path_cluster_leaf_table = 'analysis_data/leaf_information.tsv'
# 前処理済みのjsonl
path_preprocessed_jsonl = './text/processed_data.jsonl'

cluster_leaf_table = load_leaf_table(path_cluster_leaf_table)
# クラスタ数
n_cluster = len(set([r[2] for r in cluster_leaf_table]))
logging.info(f'クラスタ数 -> {n_cluster}')

# クラスタの大きさ分布
cluster_distribution = [[r[1] for r in g_obj]
                        for custer_id, g_obj
                        in itertools.groupby(sorted(cluster_leaf_table, key=lambda t: t[2]), key=lambda t: t[2])]
clsuter_distribution = [len(set(l)) for l in cluster_distribution]
min_cluster_per_cluster = min(clsuter_distribution)
max_cluster_per_cluster = max(clsuter_distribution)
avg_cluster_per_cluster = statistics.mean(clsuter_distribution)
median_cluster_per_cluster = statistics.median(clsuter_distribution)
logging.info(f'クラスタの統計 最小値:{min_cluster_per_cluster} 最大値:{max_cluster_per_cluster} \
平均:{avg_cluster_per_cluster} 中央値:{median_cluster_per_cluster}')

# クラスタごとのライブドアラベル分布
livedoor_label_distribution = [[r[3] for r in g_obj]
                               for custer_id, g_obj
                               in itertools.groupby(sorted(cluster_leaf_table, key=lambda t: t[2]), key=lambda t: t[2])]
logging.info(f'ライブドアラベル分布のリスト: {livedoor_label_distribution[:10]}')
# クラスタごとのラベル偏りを数値化してみる
numeric_livedoor_label_distribution = [len(set(l)) for l in livedoor_label_distribution]
min_label_per_cluster = min(numeric_livedoor_label_distribution)
max_label_per_cluster = max(numeric_livedoor_label_distribution)
avg_label_per_cluster = statistics.mean(numeric_livedoor_label_distribution)
median_label_per_cluster = statistics.median(numeric_livedoor_label_distribution)
logging.info(f'ライブドアラベル分布の統計 最小値:{min_label_per_cluster} 最大値:{max_label_per_cluster} \
平均:{avg_label_per_cluster} 中央値:{median_label_per_cluster}')


# クラスタごとの特徴的な単語を探索する（特徴量重み付け）
# こんな形の入力に変形したい
sample_input_dict = {
    "label_a": [
        ["I", "aa", "aa", "aa", "aa", "aa"],
        ["bb", "aa", "aa", "aa", "aa", "aa"],
        ["I", "aa", "hero", "some", "ok", "aa"]
    ],
    "label_b": [
        ["bb", "bb", "bb"],
        ["bb", "bb", "bb"],
        ["hero", "ok", "bb"],
        ["hero", "cc", "bb"],
    ],
    "label_c": [
        ["cc", "cc", "cc"],
        ["cc", "cc", "bb"],
        ["xx", "xx", "cc"],
        ["aa", "xx", "cc"],
    ]
}

# 前処理済みデータから{file_name: [単語]}のdictを得る
filename2morphs = load_preprocessed_record(path_preprocessed_jsonl)
# テーブル情報から[(クラスタ番号, 元ファイル名)]を得る
arg_information = [(r[2], json.loads(r[4])) for r in cluster_leaf_table]
# [(クラスタ番号, [単語])]のリストを作る
cluster_word = [(t[0], [word_pos[0] for word_pos in filename2morphs[t[1]['file_name']]]) for t in arg_information]
# 入力形式を整える
input_dict = {c_id: [t[1] for t in g_obj]
              for c_id, g_obj
              in itertools.groupby(sorted(cluster_word, key=lambda t: t[0]), key=lambda t: t[0])}
feature_selection_result = interface.run_feature_selection(input_dict, method='tf_idf', use_cython=True).convert_score_matrix2score_record()
# 重み付け結果をファイル出力
import pandas
df_feature_selection = pandas.DataFrame(feature_selection_result)
df_feature_selection.to_csv('./analysis_data/feature_selection.csv', index_label=False, index=False)
