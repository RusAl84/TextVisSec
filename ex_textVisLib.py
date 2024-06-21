import yellowbrick
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

#Counter({'ham': 4827, 'spam': 747})
import collections

with open('./smsspamcollection/SMSSpamCollection', encoding="UTF-8") as f:
    data = [line.strip().split('\t') for line in f.readlines()]

y, text = zip(*data)

print(collections.Counter(y)) #Counter({'ham': 4827, 'spam': 747})



#Token Frequency Distribution
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vec = CountVectorizer(stop_words="english")

transformed_data = vec.fit_transform(text)
from yellowbrick.text import FreqDistVisualizer

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)

freq_dist_viz = FreqDistVisualizer(features=vec.get_feature_names_out(), color="tomato", n=30, ax=ax)

freq_dist_viz.fit(transformed_data)

freq_dist_viz.show();

from yellowbrick.text import freqdist

fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(111)

freqdist(vec.get_feature_names_out(), transformed_data, orient="v", color="lime", ax=ax);


#t-SNE Corpus Visualization

from yellowbrick.text import TSNEVisualizer
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(stop_words="english")
transformed_text = vec.fit_transform(text)

fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111)

tsne_viz = TSNEVisualizer(ax=ax,
                        decompose="svd",
                        decompose_by=50,
                        colors=["tomato", "lime"],
                        random_state=123)

tsne_viz.fit(transformed_text.toarray(), y)

tsne_viz.show();


fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111)

tsne_viz = TSNEVisualizer(ax=ax,
                        decompose="pca",
                        decompose_by=50,
                        colors=["tomato", "lime"],
                        random_state=123)

tsne_viz.fit(transformed_text.toarray(), y)

tsne_viz.show();


from yellowbrick.text.tsne import tsne

vec = TfidfVectorizer(stop_words="english")
transformed_text = vec.fit_transform(text)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)

tsne(transformed_text.toarray(), y, ax=ax, decompose="pca", decompose_by=100, colors=["dodgerblue", "fuchsia"]);


#
from yellowbrick.text import DispersionPlot

total_docs = [doc.split() for doc in text]

target_words = ["free", "download", "win", "congrats", "crazy", "customer"]

fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(111)

visualizer = DispersionPlot(target_words,
                            ignore_case=True,
                            color=["lime", "tomato"],
                            ax=ax)
visualizer.fit(total_docs, y)
visualizer.show();

#
from yellowbrick.text import dispersion

target_words = ["free", "download", "win", "congrats", "crazy", "customer"]

fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(111)

dispersion(target_words, total_docs, y=y, ax=ax, ignore_case=False, color=["lawngreen", "red"]);