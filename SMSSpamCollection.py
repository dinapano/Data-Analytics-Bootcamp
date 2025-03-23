"""This project aims to develop a Spam Detection Model for SMS messages, utilizing Transfer Learning with DistilBERT for
improved accuracy. The focus is on creating an end-to-end solution, including preprocessing, model training,
and prediction generation. Performance evaluation metrics will be used to assess the model's effectiveness, and the
trained model will be saved for future deployment."""

import tensorflow as tf
from tensorflow.keras import activations, losses
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import pandas as pd
import pickle

# Reading the SMS spam collection dataset from a CSV file into a Pandas DataFrame
# The 'sep' parameter specifies that the data is separated by tabs ('\t')
# The 'names' parameter assigns names to the columns of the DataFrame
df = pd.read_csv('Workearly/SMSSpamCollection', sep='\t', names=["label", "message"])
# Printing the shape of the DataFrame
print(df.shape)
# Mapping the labels 'ham' and 'spam' to numeric values 0 and 1, respectively
# This is done to convert the labels into a format suitable for machine learning models
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
# Displaying the first few rows of the DataFrame to verify the changes
print(df.head())
# Extracting the 'message' column from the DataFrame and converting it to a list
x = list(df['message'])
# Extracting the 'label' column from the DataFrame and converting it to a list
y = list(df['label'])

# Define the model name and maximum length of tokens
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 20
# Select the first message from the dataset
message = x[0]
# Load the tokenizer for the specified DistilBERT model
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
# Tokenize the message using the tokenizer, ensuring it has a maximum length of MAX_LEN
inputs = tokenizer(message, max_length=MAX_LEN, truncation=True, padding=True)
# Print the original message, tokenized input IDs, and attention mask
print(f'message: \'{message}\'')
print(f'input ids: {inputs["input_ids"]}')
print(f'attention mask: {inputs["attention_mask"]}')


def construct_encodings(x, tkzr, max_len, truncation=True, padding=True):
    """
    Function to construct encodings for input sequences using a tokenizer.

    Args:
    - x: Input sequences to be encoded.
    - tkzr: Tokenizer object used for encoding.
    - max_len: Maximum length of the encoded sequences.
    - trucation: Whether to truncate sequences to `max_len`.
    - padding: Whether to pad sequences to `max_len`.

    Returns:
    - Encodings: Encoded representations of the input sequences.

    """
    # Use the tokenizer to encode the input sequences
    encodings = tkzr(
        x,
        max_length=max_len,  # Set the maximum length of the encoded sequences
        truncation=truncation,  # Truncate sequences if they exceed `max_len`
        padding=padding  # Pad sequences if they are shorter than `max_len`
    )
    return encodings


# Call the construct_encodings function to encode input sequences
encodings = construct_encodings(x, tokenizer, max_len=MAX_LEN)


# Define a function to construct a TensorFlow Dataset from token encodings and labels (if provided)
def construct_tfdataset(encodings, y=None):
    # If labels are provided (during training or evaluation)
    if y:
        # Create a TensorFlow Dataset from token encodings and labels
        # by slicing the encodings dictionary and combining it with the labels
        return tf.data.Dataset.from_tensor_slices((dict(encodings), y))
    else:
        # If labels are not provided (during inference/prediction)
        # This case is used when making predictions on unseen samples after training
        # Create a TensorFlow Dataset only from token encodings
        return tf.data.Dataset.from_tensor_slices(dict(encodings))


# Call the construct_tfdataset function with token encodings and labels (if available)
tfdataset = construct_tfdataset(encodings, y)

# Define the ratio of the dataset to be used for testing
TEST_SPLIT = 0.2
# Define the batch size for training and testing data
BATCH_SIZE = 32
# Calculate the size of the training set based on the test split ratio
train_size = int(len(x) * (1 - TEST_SPLIT))
# Shuffle the dataset to ensure randomness in training and testing samples
tfdataset = tfdataset.shuffle(len(x))
# Split the dataset into training and testing sets based on the calculated size
tfdataset_train = tfdataset.take(train_size)  # Take the first `train_size` samples for training
tfdataset_test = tfdataset.skip(train_size)  # Skip the first `train_size` samples for testing
# Batch the training and testing datasets using the specified batch size
tfdataset_train = tfdataset_train.batch(BATCH_SIZE)  # Batch the training set
tfdataset_test = tfdataset_test.batch(BATCH_SIZE)  # Batch the testing set

# Define the number of epochs for training
N_EPOCHS = 5
# Load the pre-trained DistilBERT model for sequence classification
model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
# Define the loss function for the model
loss = losses.SparseCategoricalCrossentropy(from_logits=True)
# Compile the model with the specified loss function and evaluation metric
model.compile(loss=loss, metrics=['accuracy'])
# Train the model on the training dataset
model.fit(tfdataset_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS)

# Evaluate the trained model on the test dataset
# The model.evaluate() function computes performance metrics such as loss and accuracy
# It takes the test dataset (tfdataset_test) as input and returns evaluation results
# We set return_dict=True to return evaluation results as a dictionary
# The batch_size parameter specifies the number of samples processed per batch during evaluation
benchmarks = model.evaluate(tfdataset_test, return_dict=True, batch_size=BATCH_SIZE)
# Print the evaluation results
print(benchmarks)


# Define a function to create a predictor for text classification
def create_predictor(model, model_name, max_len):
    # Load the tokenizer for the specified pre-trained DistilBERT model
    tkzr = DistilBertTokenizer.from_pretrained(model_name)

    # Define a nested function to predict probabilities for text input
    def predict_proba(text):
        # Preprocess the text input
        x = [text]
        # Construct input encodings using the tokenizer
        encodings = construct_encodings(x, tkzr, max_len=max_len)
        # Construct a TensorFlow dataset from the input encodings
        tfdataset = construct_tfdataset(encodings)
        # Batch the dataset with a batch size of 1
        tfdataset = tfdataset.batch(1)
        # Use the provided model to predict logits for the text input
        preds = model.predict(tfdataset).logits
        # Apply softmax activation to convert logits to probabilities
        preds = activations.softmax(tf.convert_to_tensor(preds)).numpy()
        # Return the predicted probability of the positive class
        return preds[0][0]

    # Return the nested prediction function
    return predict_proba


# Create a predictor function using the provided model, model name, and maximum sequence length
clf = create_predictor(model, MODEL_NAME, MAX_LEN)
# Test the predictor function with a sample text input and print the predicted probability
print(clf('New Job opportunity for you in Dubai'))

# Saving the trained model to the specified directory
model.save_pretrained('./model/clf')
# Saving metadata (such as model name and maximum length) using pickle
# Here, MODEL_NAME and MAX_LEN are assumed to be variables containing relevant information
# 'wb' mode is used to write binary data to the file
with open('./model/info.pkl', 'wb') as f:
    # Dumping the tuple containing metadata into the pickle file
    pickle.dump((MODEL_NAME, MAX_LEN), f)

# Load the pre-trained DistilBERT model from the specified directory
new_model = TFDistilBertForSequenceClassification.from_pretrained('./model/clf')
# Load additional information such as the model name and maximum sequence length from a pickled file
model_name, max_len = pickle.load(open('./model/info.pkl', 'rb'))
# Create a predictor function using the loaded model, model name, and maximum sequence length
clf = create_predictor(new_model, model_name, max_len)
# Test the predictor function with a sample text input
print(clf('NEw Job opportunity for you in Dubai'))