import urllib3
import re
import json
import requests
import networkx as nx  #type:ignore
from urllib.parse import quote
import pickle

def fetch_country_musicians():
    # Initialize HTTP manager
    http = urllib3.PoolManager()

    # Fetch list of country music performers
    baseurl = "https://en.wikipedia.org/w/api.php?"
    query = f"{baseurl}action=query&titles=List_of_country_music_performers&prop=revisions&rvprop=content&format=json"

    wikiresponse = http.request('GET', query)
    wikisource = wikiresponse.data.decode('utf-8')
    wikijson = json.loads(wikisource)
    text = wikijson["query"]["pages"]
    wiki_text = json.dumps(text)

    # Extract performer names
    results = re.findall(r"\[\[(.*?)\]\]", wiki_text)
    cleaned_data = [name.replace(" ", "_") for name in results if not (name.startswith("File:") or name.startswith("Image:"))]
    list_of_singers = [name.replace("\\", "") for name in cleaned_data[1:]]

    return list_of_singers

def fetch_wiki_content(singers):
    # Initialize data structures
    performer_text = {}  # Store raw wiki content
    performer_contents = {}  # Store extracted links
    http = urllib3.PoolManager()

    print("Fetching Wikipedia content for each performer...")
    total = len(singers)

    for i, singer in enumerate(singers):
        if i % 50 == 0:  # Progress update
            print(f"Processing {i}/{total} performers...")

        baseurl = "https://en.wikipedia.org/w/api.php?"
        query = f"{baseurl}action=query&titles={quote(singer)}&prop=revisions&rvprop=content&format=json"

        try:
            response = requests.get(query, timeout=10)
            if response.status_code == 200:
                wikiresponse = http.request('GET', query)
                wikisource = wikiresponse.data.decode('utf-8')
                wikijson = json.loads(wikisource)
                text = wikijson["query"]["pages"]
                wiki_text = json.dumps(text)
                performer_text[singer] = wiki_text

                # Extract links
                links = re.findall(r"\[\[(.*?)\]\]", wiki_text)
                performer_contents[singer] = [link.replace(" ", "_").replace("\\", "") for link in links]
        except Exception as e:
            print(f"Error processing {singer}: {str(e)}")

    return performer_text, performer_contents

def build_network(performer_contents):
    # Create network from extracted links
    matches = []
    for performer, links in performer_contents.items():
        for link in links:
            if link in performer_contents:
                matches.append((performer, link))

    matches = list(set(matches))  # Remove duplicates

    # Create directed graph
    G = nx.DiGraph()
    G.add_nodes_from(performer_contents.keys())
    G.add_edges_from(matches)

    return G

def calculate_content_size(performer_text):
    content_size = []
    for performer, content in performer_text.items():
        words = re.findall(r'\w+', content)
        content_size.append((performer, len(words)))
    return content_size

def main():
    # Fetch and process data
    singers = fetch_country_musicians()
    performer_text, performer_contents = fetch_wiki_content(singers)
    G = build_network(performer_contents)
    content_size = calculate_content_size(performer_text)

    # Create undirected version
    G_undirected = G.to_undirected()

    # Save all necessary data for Week 8
    data_to_save = {
        'performer_text': performer_text,  # Raw wiki content
        'performer_contents': performer_contents,  # Extracted links
        'graph_directed': G,  # Directed network
        'graph_undirected': G_undirected,  # Undirected network
        'content_size': content_size  # Page lengths
    }

    # Save to disk
    print("Saving data to disk...")
    with open('country_music_data.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)

    # Save networks separately in different formats
    nx.write_gexf(G, "country_music_network_directed.gexf")
    nx.write_gexf(G_undirected, "country_music_network_undirected.gexf")

    # Print some basic statistics
    print("\nNetwork Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print("Data has been saved to 'country_music_data.pkl'")
    print("Networks have been saved as .gexf files")

if __name__ == "__main__":
    main()