"""In this case study, we embark on a journey into the domain of NLP with a focus on text classification,
particularly in the context of tweets. Our aim is to develop a deep learning model capable of discerning tweets related
to disasters from those that are not. By leveraging a dataset comprising tweets sourced from diverse sources and topics,
we delve into the preprocessing, analysis, and modeling of textual data. Our objectives encompass not only the
construction and optimization of a deep learning architecture but also the evaluation of its performance metrics and
potential deployment considerations. Through this exploration, we aim to demonstrate the effectiveness and adaptability
of deep learning in addressing real-world challenges in tweet classification, while providing insights into the nuances
of NLP model development and deployment.

Data Description:
id: a unique identifier for each tweet
text: the text of the tweet
location: the location the tweet was sent from (may be blank)
keyword: a particular keyword from the tweet (may be blank)
target: in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)"""

# IMPORTING LIBRARIES
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

# LOADING & INSPECTING DATASET
# Read the CSV file into a DataFrame
df = pd.read_csv('Workearly/train.csv')
# Print the first few rows of the DataFrame (head)
print("First few rows of the DataFrame:")
print(df.head())
# Print the shape of the DataFrame
print("\nDataFrame shape:")
print(df.shape)
# Print the number of entries labeled as 'Disaster' (target == 1)
print("\nNumber of entries labeled as 'Disaster':")
print((df.target == 1).sum())
# Print the number of entries labeled as 'No Disaster' (target == 0)
print("\nNumber of entries labeled as 'No Disaster':")
print((df.target == 0).sum())

# TEXT DATA PREPROCESSING
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
# Convert text in 'text' column to lowercase
df['text'] = df['text'].apply(lambda x: x.lower())
# Define preprocessing functions
def remove_URL(text):
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', text)
def remove_punct(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.isalnum()]
    return ' '.join(filtered_words)
    # Define a function to remove stopwords from text
def remove_stopwords(text):
    # Tokenize the text into words
    words = text.split()
    # Remove stopwords
    filtered_words = [word for word in words if word.lower() not in stopwords.words('english')]
    # Join the filtered words back into a string
    return ' '.join(filtered_words)
# 'text' is the column containing text data in your DataFrame 'df'
df["text"] = df.text.map(remove_URL) # map(lambda x: remove_URL(x))
df["text"] = df.text.map(remove_punct)
df["text"] = df.text.map(remove_stopwords)

# WORD FREQUENCY ANALYSIS
# Function to count unique words in a text column
def counter_word(text_col):
    # Initialize a Counter object to count word occurrences
    count = Counter()
    # Iterate over each text entry in the text column
    for text in text_col.values:
        # Split the text into words and update the counter
        for word in text.split():
            count[word] += 1
    return count
# Call the counter_word function with the 'text' column of DataFrame 'df'
counter = counter_word(df['text'])
# Print the total number of unique words
print("Total number of unique words:", len(counter))
# Print the resulting word count dictionary
print("\nCounter:", counter)
# Print the most common words and their frequencies
print("\nMost Common words:", counter.most_common(5))


# SPLITTING DATAFRAME INTO TRAINING AND VALIDATION SETS
# Define features (text) and labels
X = df['text'].values
y = df['target'].values
# Split the dataset into training and validation sets
train_sentences, val_sentences, train_labels, val_labels = train_test_split(X,
                                                                            y,
                                                                            test_size=0.2,
                                                                            random_state=42)
train_sentences.shape, val_sentences.shape


# TOKENIZATION AND WORD INDEX CREATION
# Calculate the number of unique words from the counter
num_unique_words = len(counter)
# Initialize the Tokenizer with the number of unique words
tokenizer = Tokenizer(num_words=num_unique_words)
# Fit the Tokenizer on the training sentences
tokenizer.fit_on_texts(train_sentences)
# Obtain the word index dictionary from the Tokenizer
word_index = tokenizer.word_index
# Print the word index dictionary
print(word_index)


# TOKENIZE TEXT DATA IN THE TRAINING AND VALIDATION SETS
train_sequences = tokenizer.texts_to_sequences(train_sentences)
val_sequences = tokenizer.texts_to_sequences(val_sentences)
# Print a subset of original sentences from the training set
print("Subset of original sentences from the training set:")
print(train_sentences[10:15])
# Print the corresponding sequences of integers for the subset of sentences
print("\nCorresponding sequences of integers:")
print(train_sequences[10:15])


# SEQUENCE PADDING FOR TEXT SEQUENCES
# Define the maximum length for sequences
max_length = 20
# Pad sequences for training and validation sets
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding="post", truncating="post")
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding="post", truncating="post")
# Print the shapes of padded sequences
print("Shape of padded training sequences:", train_padded.shape)
print("Shape of padded validation sequences:", val_padded.shape)
# Print an example of a padded training sequence
print("\nExample of a padded training sequence:")
print(train_padded[10])
# Print corresponding original sentence, its sequence, and padded sequence
print("\nOriginal sentence:")
print(train_sentences[10])
print("\nSequence representation:")
print(train_sequences[10])
print("\nPadded sequence:")
print(train_padded[10])


# MAPPING INDICES TO WORDS
# Check reversing the indices
# Create a dictionary to map integer indices to words
reverse_word_index = dict([(idx, word) for (word, idx) in word_index.items()])
# Print the reversed word index
print("Reversed word index:", reverse_word_index)


# DECODING INTEGER INDICES TO TEXT
def decode(sequence):
    # Use list comprehension to map integer indices to words using reverse_word_index dictionary
    # If the index is not found in the reverse_word_index, replace it with "?"
    return " ".join([reverse_word_index.get(idx, "?") for idx in sequence])
# Decode the sequence of indices back to text
decoded_text = decode(train_sequences[10])
# Print the original sequence of indices and the decoded text
print("Original sequence of indices:")
print(train_sequences[10])
print("\nDecoded text:")
print(decoded_text)

# SEQUENTIAL NEURAL NETWORK MODEL
# Create a Sequential model
model = keras.models.Sequential()
# Add an Embedding layer to the model
model.add(layers.Embedding(num_unique_words, 32, input_length=max_length))
# Add an LSTM layer to the model
model.add(layers.LSTM(64, dropout=0.1))
# Add a Dense layer to the model
model.add(layers.Dense(1, activation="sigmoid"))
# Print a summary of the model architecture
model.summary()

# MODEL COMPILATION
# Define the loss function
loss = keras.losses.BinaryCrossentropy(from_logits=False)
# Define the optimizer
optim = keras.optimizers.Adam(learning_rate=0.001)
# Define evaluation metrics
metrics = ["accuracy"]
# Compile the model
model.compile(loss=loss, optimizer=optim, metrics=metrics)


# TRAINING THE MODEL
model.fit(
    train_padded,            # Training data: padded sequences
    train_labels,            # Training labels
    epochs=20,               # Number of training epochs
    validation_data=(val_padded, val_labels),  # Validation data for evaluation during training
    verbose=2       # Verbosity mode: 0 (silent), 1 (progress bar), 2 (one line per epoch)
)


# MAKING BINARY PREDICTIONS FROM MODEL PROBABILITIES
# Make predictions using the trained model on the padded training sequences
predictions = model.predict(train_padded)
# Convert predicted probabilities to binary labels using a threshold of 0.5
predictions = [1 if p > 0.5 else 0 for p in predictions]



# INSPECTING TRAINING DATA, TRUE LABELS, AND PREDICTIONS MAKING BINARY PREDICTIONS
# FROM MODEL PROBABILITIES
# Print a subset of training sentences to inspect the original text data
print("Subset of Training Sentences:")
print(train_sentences[:10])
# Print a subset of true labels to inspect the ground truth
print("\nSubset of True Labels:")
print(train_labels[:10])
# Print a subset of predicted labels to inspect the model's binary predictions
print("\nSubset of Predicted Labels:")
print(predictions[:10])