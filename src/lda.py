import csv
from gensim import corpora, models


# Function to load topics from a CSV file
def load_csv_topics(csv_filepath):
    topics = []
    with open(csv_filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            topics.append(row[0])  # The first column is the topic name
    return topics


# Function to map LDA topics to CSV topics
def map_lda_to_csv_topics(lda_topics, csv_topics):
    # In a real application, you would use some sort of similarity measure to map the LDA topics to your CSV topics.
    # Here we're just assigning the LDA topic index to a CSV topic based on the index.
    print("lda", lda_topics)
    print("csv", csv_topics)
    num_csv_topics = len(csv_topics)
    mapped_topics = {}

    for img_name, lda_topic_index in lda_topics.items():
        # Map the LDA topic index to a CSV topic by using modulo to loop around the CSV topic list
        mapped_topic = csv_topics[lda_topic_index % num_csv_topics]
        mapped_topics[img_name] = mapped_topic

    return mapped_topics


def run_lda(top_elements, csv_filepath='data/topics.csv', num_topics=100):
    # Load CSV topics
    csv_topics = load_csv_topics(csv_filepath)
    # Flatten the recognized elements into a list of lists for LDA
    texts = [[element for element, _ in elements] for elements in top_elements.values()]

    # Create a dictionary and corpus for LDA
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Run LDA
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=100,
                                update_every=1, chunksize=100, alpha='auto', per_word_topics=True)

    # Determine the most probable topic for each image
    image_lda_topics = {}
    for img_name, bow in zip(top_elements.keys(), corpus):
        # Get the topic distribution for the image
        img_topics = lda_model.get_document_topics(bow)
        # Sort topics by probability
        img_topics = sorted(img_topics, key=lambda x: x[1], reverse=True)
        # Take the most probable topic
        image_lda_topics[img_name] = img_topics[0][0] if img_topics else None

    # Map LDA topics to CSV topics
    image_topics = map_lda_to_csv_topics(image_lda_topics, csv_topics)

    return image_topics, lda_model
