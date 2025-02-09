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

model = tf.keras.models.load_model("shakespeare_model.keras")

def generate_text(model, start_string, num_generate=500):
    # Create an input with batch size 64
    input_ids = [char2idx[s] for s in start_string]
    # We need to expand dims and then tile to create a batch of 64 identical sequences
    input_eval = tf.expand_dims(input_ids, 0)            # shape: (1, len(start_string))
    input_eval = tf.tile(input_eval, [64, 1])              # shape: (64, len(start_string))

    text_generated = []
    # Reset the LSTM states by calling reset_states on the stateful layer directly
    model.layers[1].reset_states()

    for _ in range(num_generate):
        predictions = model(input_eval)  # shape: (64, seq_len, vocab_size)
        # Select predictions for the first sample in the batch and only the last time step
        predictions = predictions[0, -1, :]  # shape: (vocab_size,)

        predicted_id = tf.random.categorical(tf.expand_dims(predictions, 0), num_samples=1)[0, 0].numpy()

        # Prepare input for next step: tile predicted character to match batch size 64
        input_eval = tf.tile(tf.expand_dims([predicted_id], 0), [64, 1])
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

print(generate_text(model, start_string="T", num_generate=500))



