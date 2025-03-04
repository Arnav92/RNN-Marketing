import numpy as np
import re
import string
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam

############################################
# 1. Example Marketing Data (manually added)
############################################
marketing_slogans = [
    "Save big on your next purchase",
    "Experience the comfort you deserve",
    "Unlock your potential with our new course",
    "The best coffee in town",
    "Shop now and enjoy exclusive deals",
    "Get more for less with our discount sale",
    "Your dream vacation awaits you",
    "Upgrade your style with our latest collection",
    "Feel the difference with our organic ingredients",
    "Take your business to the next level with us",
    "Transform your home into a luxury retreat",
    "Discover flavors that excite your taste buds"
]

############################################
# 2. Basic Cleaning
############################################
def basic_clean(text):
    text = text.lower()

    # remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

    # strip extra spaces
    text = re.sub("\s+", " ", text).strip()
    return text

cleaned_slogans = [basic_clean(slogan) for slogan in marketing_slogans]

############################################
# 3. Create Sequences for Next-Word Prediction
############################################
tokenizer = Tokenizer()
tokenizer.fit_on_texts(cleaned_slogans)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for slogan in cleaned_slogans:
    token_list = tokenizer.texts_to_sequences([slogan])[0]

    # create n-gram sequences
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i+1]
        input_sequences.append(n_gram_seq)

# Find max sequence length
max_seq_len = max(len(x) for x in input_sequences)

# Pad sequences
padded_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

# Separate predictors and label
X = padded_sequences[:, :-1]
y = padded_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

############################################
# 4. Build a Simple LSTM Model
############################################
model = Sequential()
model.add(Embedding(input_dim=total_words,
                    output_dim=64,
                    input_length=max_seq_len - 1))
model.add(LSTM(128))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.01),
              metrics=['accuracy'])

print(model.summary())

############################################
# 5. Train the Model (Short Demo)
############################################
history = model.fit(X, y, epochs=50, verbose=1)  # set verbose=1 if you want logs

############################################
# 6. Slogan Generation with Sampling
############################################
def sample_with_top_k(predictions, top_k=5, no_repeat=None):
    """
    Sample from the top_k most probable words.
    If no_repeat is given (e.g., last token), we remove that token from consideration to reduce repetition.
    """
    predictions = np.squeeze(predictions)  # shape -> (vocab_size,)
    # Zero out any 'no_repeat' token
    if no_repeat is not None:
        predictions[no_repeat] = 0

    # Get top_k indices
    top_k_indices = predictions.argsort()[-top_k:]
    top_k_probs = predictions[top_k_indices]

    # Convert to probabilities (softmax over top_k)
    top_k_probs = top_k_probs / np.sum(top_k_probs)

    # Random choice among top_k
    chosen_index = np.random.choice(top_k_indices, p=top_k_probs)
    return chosen_index

def generate_slogan(seed_text, next_words=5, top_k=5):
    """
    Generate new text by predicting the next word repeatedly,
    sampling from the model's top_k predictions to avoid repetition.
    """
    result = seed_text
    last_word_index = None

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([result])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')

        predictions = model.predict(token_list, verbose=0)[0]  # shape (vocab_size,)

        # Sample from top_k
        predicted_index = sample_with_top_k(predictions, top_k=top_k, no_repeat=last_word_index)

        # Convert index to word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break

        # Append the new word
        result += " " + output_word
        # Keep track of the last word index to reduce consecutive repetition
        last_word_index = predicted_index

    return result

############################################
# 7. Test with Various Seed Texts
############################################
test_prompts = [
    "The best chocolate",
    "Watch the funniest movie",
    "Save big",
    "Transform your car"
]

for prompt in test_prompts:
    gen_text = generate_slogan(prompt, next_words=8, top_k=5)
    print(f"Seed text: {prompt}")
    print(f"Generated slogan: {gen_text}\n")