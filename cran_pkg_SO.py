import pandas as pd
from nltk import word_tokenize
import string
import re
import matplotlib.pyplot as plt
questions = pd.read_csv(
    "../rquestions/Questions.csv", encoding='latin1')
#.query('Id % 1000 == 0')
punc_set = set(string.punctuation) - (set(['_', '-']))
replace_punctuation = str.maketrans(''.join(punc_set), ' ' * len(punc_set))

# import html2text
# questions['Body'] = questions["Body"].apply(
#     lambda x: html2text.html2text(x).lower().translate(replace_punctuation))
regex_pat = re.compile(r'<[^<]+?>', flags=re.IGNORECASE)
questions['Body'] = questions["Body"].str.replace(
    regex_pat, ' ').str.lower().str.translate(replace_punctuation)

r_pkgs = pd.read_csv('gentoo_r_overlay.csv')
all_pkg = set(r_pkgs['name'].str.lower())


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

cv = CountVectorizer(tokenizer=word_tokenize, stop_words=None,
                     ngram_range=(1, 1), vocabulary=all_pkg, max_features=4000, token_pattern=r'(?u)\b[-a-zA-Z0-9_]+\b', min_df=1, max_df=.95)
dt_mat = cv.fit_transform(questions['Body'].fillna(" "))

tfidf_transformer = TfidfTransformer()
tfidf_mat = tfidf_transformer.fit_transform(dt_mat)
# unigrams = pd.DataFrame(dt_mat.todense(), index=questions.index,
#                         columns=cv.get_feature_names())


def plot_top(df_top_N, col_x, col_y):
    labels = [i for i in df_top_N[col_x]]
    values = [i for i in df_top_N[col_y]]
    xs = np.arange(len(labels))
    width = .85
    plt.figure(figsize=(18, 9))
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    plt.xticks(rotation=80)
    plt.xlabel('Topics')
    plt.ylabel('%s of Questions Mentioned' % col_y)
    plt.title('Bar Plot of Topics Mentioned')
    plt.bar(xs, values, width, align='center')
    plt.xticks(xs, labels)
    # plt.show()
    plt.savefig('%s_%s.png' % (col_x, col_y), dpi=200)
    plt.clf()
    plt.cla()
    plt.close()


import numpy as np
# https://stackoverflow.com/questions/33181846/programmatically-convert-pandas-dataframe-to-markdown-table
from tabulate import tabulate
occ = np.asarray(dt_mat.sum(axis=0)).ravel().tolist()
counts_df = pd.DataFrame({'term': cv.get_feature_names(), 'occurrences': occ})
counts_df_top_100 = counts_df.sort_values(
    by='occurrences', ascending=False).head(100)

plot_top(counts_df_top_100, 'term', 'occurrences')
print(tabulate(counts_df_top_100.set_index(
    "term"), tablefmt="markdown", headers="keys"))


weights = np.asarray(tfidf_mat.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cv.get_feature_names(), 'weight': weights})
weights_df_top_100 = weights_df.sort_values(
    by='weight', ascending=False).head(100)
plot_top(weights_df_top_100, 'term', 'weight')
print(tabulate(weights_df_top_100.set_index(
    "term"), tablefmt="markdown", headers="keys"))

#unigrams['year'] = questions.year
# print(bigrams)
