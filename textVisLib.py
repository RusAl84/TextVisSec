import yellowbrick
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import collections
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from yellowbrick.text import TSNEVisualizer
from sklearn.feature_extraction.text import TfidfVectorizer
from yellowbrick.text.tsne import tsne


def load_data(filename='risk.txt'):
    # filename='./smsspamcollection/SMSSpamCollection'
    with open(filename, encoding="UTF-8") as f:
        data = [line.strip().split('\t') for line in f.readlines()]
    y, text = zip(*data)
    print(collections.Counter(y)) #Counter({'ham': 4827, 'spam': 747})
    return (y, text)


#Token Frequency Distribution
def TokenFrequencyDistribution(y, text):   
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
def tSNEV_CV_SVD(y, text):
    vec = TfidfVectorizer(stop_words="english")
    transformed_text = vec.fit_transform(text)

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111)

    tsne_viz = TSNEVisualizer(ax=ax,
                        decompose="svd",
                        decompose_by=45,
                        colors=["tomato", "lime","dodgerblue", "fuchsia"],
                        random_state=123)
    tsne_viz.fit(transformed_text.toarray(), y)
    tsne_viz.show();


def tSNEV_CV_PCA(y, text):
    vec = TfidfVectorizer(stop_words="english")
    transformed_text = vec.fit_transform(text)
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111)

    tsne_viz = TSNEVisualizer(ax=ax,
                        decompose="pca",
                        decompose_by=45,
                        colors=["tomato", "lime","dodgerblue", "fuchsia"],
                        random_state=123)

    tsne_viz.fit(transformed_text.toarray(), y)
    tsne_viz.show();


def tSNEV_CV_PCA2(y, text):
    vec = TfidfVectorizer(stop_words="english")
    transformed_text = vec.fit_transform(text)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)

    tsne(transformed_text.toarray(), y, ax=ax, decompose="pca", decompose_by=45, colors=["tomato", "lime","dodgerblue", "fuchsia"]);

if __name__ == '__main__':
    # filename='risk.txt'
    # (y, text) = load_data(filename)
    # TokenFrequencyDistribution(y, text)
    # tSNEV_CV_SVD(y, text)
    # tSNEV_CV_PCA(y, text)
    # tSNEV_CV_PCA2(y, text)
    
    filename='bdu.txt'
    (y, text) = load_data(filename)
    # TokenFrequencyDistribution(y, text)
    tSNEV_CV_SVD(y, text)
    # tSNEV_CV_PCA(y, text)
    # tSNEV_CV_PCA2(y, text)