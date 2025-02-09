import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

# Load Tiny Shakespeare dataset (fix: remove `as_supervised=True`)
dataset_name = "tiny_shakespeare"
data, info = tfds.load(dataset_name, with_info=True)

# Extract text data
text = ""
for example in data['train']:  # No (input, label) pairs, just a single text feature
    text += example['text'].numpy().decode('utf-8') + "\n"

# Unique characters in the dataset
vocab = sorted(set(text))
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = np.array(vocab)

# Convert text to numerical representation
text_as_int = np.array([char2idx[c] for c in text])

# Print dataset stats
print(f"Total characters: {len(text)}")
print(f"Unique characters: {len(vocab)}")

# Sequence length for training
seq_length = 100  # Number of characters per training sample
batch_size = 64
buffer_size = 10000

# Create input-output sequences
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

# Function to split sequences into input (X) and target (y)
def split_input_target(seq):
    input_text = seq[:-1]  # All but last character
    target_text = seq[1:]  # All but first character
    return input_text, target_text

dataset = sequences.map(split_input_target)
dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

# Define model parameters
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=None),
    tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
])

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# Training epochs
epochs = 20

# Train the model
history = model.fit(dataset, epochs=epochs)

# Save the trained model
model.save("shakespeare_model.keras")
