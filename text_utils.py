"""
Text processing utilities for the Multimodal Sentiment Analysis project.

Contains tokenizer creation and text preprocessing functions.
Important: Tokenizers should be created ONCE and reused for all data splits
to avoid vocabulary inconsistency issues.
"""

import tensorflow as tf
from transformers import BertTokenizer, GPT2Tokenizer
import pickle

from config import MAX_TEXT_LENGTH, VOCAB_SIZE, TOKENIZER_PATH


# ==============================================================================
# TOKENIZER CREATION
# These functions create tokenizer instances - call them once and reuse!
# ==============================================================================

def create_bert_tokenizer():
    """
    Create and return a BERT tokenizer.
    
    Returns:
        BertTokenizer instance
    """
    return BertTokenizer.from_pretrained('bert-base-uncased')


def create_gpt2_tokenizer():
    """
    Create and return a GPT-2 tokenizer with proper padding token.
    
    Returns:
        GPT2Tokenizer instance with pad_token set
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # GPT-2 doesn't have a pad token by default, use EOS token
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def create_glove_tokenizer(texts, num_words=VOCAB_SIZE):
    """
    Create and fit a Keras tokenizer for GloVe-style embeddings.
    
    IMPORTANT: This should be fit ONLY on training data, then reused
    for validation and test data to prevent data leakage.
    
    Args:
        texts: List of text strings to fit the tokenizer on
        num_words: Maximum vocabulary size
        
    Returns:
        Fitted Keras Tokenizer instance
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def save_glove_tokenizer(tokenizer, path=TOKENIZER_PATH):
    """
    Save a Keras tokenizer to disk.
    
    Args:
        tokenizer: Fitted Keras Tokenizer
        path: File path to save to
    """
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {path}")


def load_glove_tokenizer(path=TOKENIZER_PATH):
    """
    Load a Keras tokenizer from disk.
    
    Args:
        path: File path to load from
        
    Returns:
        Loaded Keras Tokenizer
    """
    with open(path, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer loaded from {path}")
    return tokenizer


# ==============================================================================
# TEXT TOKENIZATION FUNCTIONS
# ==============================================================================

def tokenize_for_bert(texts, tokenizer, max_length=MAX_TEXT_LENGTH):
    """
    Tokenize texts for BERT model (IFFSA).
    
    Args:
        texts: List of text strings
        tokenizer: BERT tokenizer instance
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (input_ids, attention_mask) as TensorFlow tensors
    """
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='tf'
    )
    return encodings['input_ids'], encodings['attention_mask']


def tokenize_for_gpt2(texts, tokenizer, max_length=MAX_TEXT_LENGTH):
    """
    Tokenize texts for GPT-2 model (BFSA).
    
    Args:
        texts: List of text strings
        tokenizer: GPT-2 tokenizer instance
        max_length: Maximum sequence length
        
    Returns:
        input_ids as TensorFlow tensor
    """
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='tf'
    )
    return encodings['input_ids']


def tokenize_for_glove(texts, tokenizer, max_length=MAX_TEXT_LENGTH):
    """
    Tokenize texts for GloVe/LSTM model (TBJE).
    
    IMPORTANT: Use a tokenizer that was fit on training data only.
    
    Args:
        texts: List of text strings
        tokenizer: Fitted Keras Tokenizer instance
        max_length: Maximum sequence length
        
    Returns:
        Padded sequences as numpy array
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, 
        maxlen=max_length,
        padding='post',
        truncating='post'
    )
    return padded
