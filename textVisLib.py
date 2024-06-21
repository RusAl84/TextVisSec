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

freq_dist_viz = FreqDistVisualizer(features=vec.get_feature_names(), color="tomato", n=30, ax=ax)

freq_dist_viz.fit(transformed_data)

freq_dist_viz.show();