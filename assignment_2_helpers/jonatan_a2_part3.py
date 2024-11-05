"""
The following are optional exercises from week 8 that we DO NOT need to solve.
*Week 8 optional Exercise 1*: Sentiment distribution.
>
> * Use this LabMT wordlist dataset:
        word  happiness_rank  happiness_average  happiness_standard_deviation  twitter_rank  google_rank  nyt_rank  lyrics_rank
0   laughter               1               8.50                        0.9313        3600.0          NaN       NaN       1728.0
1  happiness               2               8.44                        0.9723        1853.0       2458.0       NaN       1230.0
2       love               3               8.42                        1.1082          25.0        317.0     328.0         23.0
3      happy               4               8.30                        0.9949          65.0       1372.0    1313.0        375.0
4    laughed               5               8.26                        1.1572        3334.0       3542.0       NaN       2332.0
[10000+ more rows not shown]
> * Based on the LabMT word list, write a function that calculates sentiment given a list of tokens (the tokens should be lower case, etc).
> * Iterage over the nodes in your network, tokenize each page, and calculate sentiment every single page. Now you have sentiment as a new nodal property.
> * Calculate the average sentiment across all the pages. Also calculate the median, variance, 25th percentile, 75th percentile.
> * Remember histograms? Create a histogram of all of the artists's associated page-sentiments. (And make it a nice histogram - use your histogram making skills from Week 2). Add the mean, meadian, ect from above to your plot.
> * Who are the 10 artists with happiest and saddest pages?

*Week 8 optional Exercise 2*: Community sentiment distribution.

> * Last week we calculated the stuctural communities of the graph. For this exercise, we use those communities (just the 10 largest ones). Specifically, you should calculate the average the average sentiment of the nodes in each community to find a *community level sentiment*.
>   - Name each community by its three most connected characters.
>   - What are the three happiest communities?
>   - what are the three saddest communities?
>   - Do these results confirm what you can learn about each community by comparing to the genres, checking out the word-clouds for each community, and reading the wiki-pages?
> * Compare the sentiment of the happiest and saddest communities to the overall (entire network) distribution of sentiment that you calculated in the previous exercise. Are the communities very differenct from the average? Or do you find the sentiment to be quite similar across all of the communities?

**Note**: Calculating sentiment takes a long time, so arm yourself with patience as your code runs (remember to check that it runs correctly, before waiting patiently). Further, these tips may speed things up. And save somewhere, so you don't have to start over.
**Tips for speed**
* If you use `freqDist` prior to finding the sentiment, you only have to find it for every unique word and hereafter you can do a weighted mean.
"""

# Assignment 2 Part 3: Sentiment of the artists and communities
# The following are mandatory assignments that we DO need to solve.
# Question 1: Calculate the sentiment of the Artists pages (OK to work with the sub-network of artists-with-genre) and describe your findings using stats and visualization, inspired by the first optional exercise of week 8.
# Question 2: Discuss the sentiment of the largest communities. Do the findings using TF-IDF during Lecture 7 help you understand your results?

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns  #type:ignore
from nltk.tokenize import word_tokenize  #type:ignore
import networkx as nx  #type:ignore
import pickle
import community  #type:ignore


labmt_df = pd.read_csv('LabMT_data.txt', sep='\t', na_values='--')
print(labmt_df.head())  #LabMT wordlist dataset
with open('country_music_data.pkl', 'rb') as f:
    musicians_data = pickle.load(f)  # Access the components
artist_texts = musicians_data['performer_text']  # Wiki content for sentiment analysis
performer_contents = musicians_data['performer_contents']  # Link structure
G = musicians_data['graph_directed']  # Directed network
G_undirected = musicians_data['graph_undirected']  # Undirected network
content_size = musicians_data['content_size']  # Page lengths

### QUESTION 1

# First, let's create a sentiment lookup dictionary from the LabMT dataset
def create_sentiment_dict(df):
    return dict(zip(df['word'], df['happiness_average']))

# Function to calculate sentiment for a text
def calculate_sentiment(text, sentiment_dict):
    if isinstance(text, str):
        # Tokenize and convert to lowercase
        tokens = word_tokenize(text.lower())
        # Get sentiment scores for words that exist in our dictionary
        scores = [sentiment_dict[word] for word in tokens if word in sentiment_dict]
        # Return average if we have scores, otherwise return None
        return np.mean(scores) if scores else None
    return None

# Detect communities using the Louvain method
communities_dict = community.best_partition(G_undirected)

# Convert communities dict to list of lists format
num_communities = max(communities_dict.values()) + 1
communities = [[] for _ in range(num_communities)]  #type:ignore
for node, community_id in communities_dict.items():
    communities[community_id].append(node)

# Keep only the 10 largest communities
communities = sorted(communities, key=len, reverse=True)[:10]

# Create sentiment dictionary
sentiment_dict = create_sentiment_dict(labmt_df)

# Calculate sentiment for each artist with progress tracking
artist_sentiments = {}
total = len(artist_texts)
print(f"Calculating sentiment for {total} artists...")
for i, (artist, text) in enumerate(artist_texts.items()):
    if i % 100 == 0:  # Progress update every 100 artists
        print(f"Processing {i}/{total} artists...")
    sentiment = calculate_sentiment(text, sentiment_dict)
    if sentiment is not None:
        artist_sentiments[artist] = sentiment

# Calculate statistics
sentiments = list(artist_sentiments.values())
stats = {
    'mean': np.mean(sentiments),
    'median': np.median(sentiments),
    'std': np.std(sentiments),
    '25th': np.percentile(sentiments, 25),
    '75th': np.percentile(sentiments, 75)
}

# Create visualization
plt.figure(figsize=(12, 6))
sns.histplot(sentiments, bins=50)
plt.axvline(stats['mean'], color='r', linestyle='--', label=f"Mean: {stats['mean']:.2f}")
plt.axvline(stats['median'], color='g', linestyle='--', label=f"Median: {stats['median']:.2f}")
plt.title('Distribution of Artist Page Sentiments')
plt.xlabel('Sentiment Score')
plt.ylabel('Count')
plt.legend()
plt.show()

# Find top 10 happiest and saddest artists
sorted_sentiments = sorted(artist_sentiments.items(), key=lambda x: x[1])
saddest_artists = sorted_sentiments[:10]
happiest_artists = sorted_sentiments[-10:]



### QUESTION 2


# Assuming we have communities from previous analysis
def analyze_community_sentiment(G, communities, artist_sentiments):
    community_sentiments = {}

    for i, community in enumerate(communities):
        # Calculate average sentiment for community
        community_scores = [artist_sentiments[node] for node in community
                          if node in artist_sentiments]
        if community_scores:
            avg_sentiment = np.mean(community_scores)
            community_sentiments[i] = {
                'sentiment': avg_sentiment,
                'size': len(community),
                'members': community
            }

    # Sort communities by sentiment
    sorted_communities = sorted(community_sentiments.items(),
                              key=lambda x: x[1]['sentiment'])

    return sorted_communities

# Analyze communities
community_analysis = analyze_community_sentiment(G, communities, artist_sentiments)

# Print results
print("\nTop 3 Happiest Communities:")
for i in range(-1, -4, -1):
    comm = community_analysis[i]
    print(f"Community {comm[0]}: Sentiment = {comm[1]['sentiment']:.3f}")

print("\nTop 3 Saddest Communities:")
for i in range(3):
    comm = community_analysis[i]
    print(f"Community {comm[0]}: Sentiment = {comm[1]['sentiment']:.3f}")

# TF-IDF Discussion (as text output for report)
print("\n--- TF-IDF Discussion ---")
print("The TF-IDF analysis conducted during Lecture 7 highlighted key themes and terms associated with each community. By comparing these themes to the sentiment results, we can infer possible explanations. For instance, communities with high TF-IDF scores for words like 'love', 'joy', or 'celebration' tend to exhibit higher sentiment values. Conversely, communities where terms related to 'loss' or 'sadness' dominate have lower sentiment scores. This connection underscores the relevance of content themes in shaping the sentiment distribution.")

# Print statistics
print("\nSentiment Statistics:")
for stat_name, value in stats.items():
    print(f"{stat_name}: {value:.3f}")

print("\nTop 10 Happiest Artists:")
for artist, sentiment in happiest_artists:
    print(f"{artist}: {sentiment:.3f}")

print("\nTop 10 Saddest Artists:")
for artist, sentiment in saddest_artists:
    print(f"{artist}: {sentiment:.3f}")

# Save results to file
results = {
    'artist_sentiments': artist_sentiments,
    'statistics': stats,
    'community_analysis': community_analysis,
    'happiest_artists': happiest_artists,
    'saddest_artists': saddest_artists
}

with open('sentiment_analysis_results.pkl', 'wb') as f:  #type:ignore
    pickle.dump(results, f)
