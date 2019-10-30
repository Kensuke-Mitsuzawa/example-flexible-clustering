#! -*- coding: utf-8 -*-
import jsonlines
import flexible_clustering_tree
import logging
import gensim
from typing import Tuple, List, Optional
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy
import tqdm
from collections import Counter
logging.basicConfig(level=logging.INFO)

"""flexible_clustering_treeパッケージでクラスタリングする．
1層目はライブドアニュースのタイトルをword embeddingでベクトル化
2層目はライブドアニュースの単語ベクトル(bowベクトルをSDVで次元削減)
"""


def generate_one_document(input_obj,
                          word_emb_obj: gensim.models.KeyedVectors
                          ) -> Optional[Tuple[str, str, numpy.ndarray, List[str], List[str]]]:
    """１ドキュメント分の情報を生成する"""
    try:
        vectors = [word_emb_obj.get_vector(w) for w in input_obj['title_morphs'] if w in word_emb_obj.wv.vocab]
        if len(vectors) == 0:
            average_vector = numpy.zeros(word_emb_obj.wv.vector_size)
        else:
            average_vector = numpy.mean(vectors, axis=0)
    except Exception as e:
        logging.error(e)
        return None
    else:
        return input_obj['file_name'], input_obj['category'], average_vector, input_obj['morphs'], input_obj['title_morphs']


PATH_WORD_EMD_FILE = './text/jawiki.all_vectors.100d.txt'
PATH_INPUT_JSONL = './text/processed_data.jsonl'
PATH_OUTPUT_HTML = './text/output-normal.html'

logging.info('loading word emb file...')
word_emb_obj = gensim.models.KeyedVectors.load_word2vec_format(PATH_WORD_EMD_FILE)
logging.info('finished loading!')


title_vectors = []
document_morphs = []
title_morphs = []
livedoor_labels = []
livedoor_file_names = []

with jsonlines.open(PATH_INPUT_JSONL) as reader:
    for record in tqdm.tqdm(reader):
        t = generate_one_document(record, word_emb_obj)
        if t is None:
            continue
        else:
            livedoor_file_names.append(t[0])
            livedoor_labels.append(t[1])
            document_morphs.append(t[3])
            title_vectors.append(t[2])
            title_morphs.append(t[4])
    else:
        pass


logging.info('Making BOW matrix on documents...')
v = DictVectorizer(sparse=True)
D = [dict(Counter(d)) for d in document_morphs]
freq_matrix = v.fit_transform(D)

logging.info('Running SVD on BOW matrix...')
svd = TruncatedSVD(n_components=256, n_iter=7, random_state=42)
low_dim_matrix = svd.fit_transform(freq_matrix)

assert len(title_vectors) == len(low_dim_matrix) == len(livedoor_file_names) == len(livedoor_labels)

feature_1st_layer = flexible_clustering_tree.FeatureMatrixObject(level=0, matrix_object=numpy.array(title_vectors))
feature_2nd_layer = flexible_clustering_tree.FeatureMatrixObject(level=1, matrix_object=low_dim_matrix)


multi_matrix_obj = flexible_clustering_tree.MultiFeatureMatrixObject(
    matrix_objects=[feature_1st_layer, feature_2nd_layer],
    dict_index2label={i: label for i, label in enumerate(livedoor_labels)},
    dict_index2attributes={i: {
        'file_name': livedoor_file_names[i],
        'document_text': ''.join(document_morphs[i]),
        'title_text': ''.join(title_morphs[i])
    } for i, label in enumerate(livedoor_labels)}
)


from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
# clustering_operator_1st = flexible_clustering_tree.ClusteringOperator(level=0, n_cluster=8, instance_clustering=KMeans(n_clusters=8))
clustering_operator_1st = flexible_clustering_tree.ClusteringOperator(level=0, n_cluster=-1, instance_clustering=HDBSCAN())
clustering_operator_2nd = flexible_clustering_tree.ClusteringOperator(level=1, n_cluster=-1, instance_clustering=HDBSCAN())
multi_clustering_operator = flexible_clustering_tree.MultiClusteringOperator([clustering_operator_1st, clustering_operator_2nd])

# run flexible clustering
clustering_runner = flexible_clustering_tree.FlexibleClustering(max_depth=3)
index2cluster_no = clustering_runner.fit_transform(multi_matrix_obj, multi_clustering_operator)
html = clustering_runner.clustering_tree.to_html()

with open(PATH_OUTPUT_HTML, 'w') as f:
    f.write(html)

# labels_ attributeでクラスタ番号を取得できる．クラスタリングだけが目的ならば，この情報だけで十分
from collections import Counter
print(Counter(clustering_runner.labels_))

# 集計目的のテーブル情報を取得できる
import pandas
table_information = clustering_runner.clustering_tree.to_objects()
pandas.DataFrame(table_information['cluster_information']).to_csv('cluster_relation.tsv', sep='\t')
pandas.DataFrame(table_information['leaf_information']).to_csv('leaf_information.tsv', sep='\t')