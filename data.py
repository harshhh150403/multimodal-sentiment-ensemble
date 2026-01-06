"""
Data loading and preprocessing for the Multimodal Sentiment Analysis project.

Contains dataset loading, caching, and preprocessing functions.
"""

import os
import numpy as np
import gc
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

from config import (
    TRANSCRIPT_PATH, VIDEO_PATH, CACHE_DIR, MAX_FRAMES,
    TEST_SPLIT, RANDOM_SEED
)
from video_utils import extract_video_frames, visualize_video_frames


# Create a single shared analyzer instance (not recreated for each call)
_sentiment_analyzer = None

def get_sentiment_analyzer():
    """
    Get or create the shared SentimentIntensityAnalyzer instance.
    This avoids recreating the analyzer for every text sample.
    """
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentIntensityAnalyzer()
    return _sentiment_analyzer


def clear_memory():
    """Clear memory to prevent OOM errors."""
    gc.collect()
    tf.keras.backend.clear_session()


def extract_sentiment_label(text):
    """
    Extract binary sentiment label from text using VADER.
    
    Args:
        text: Input text string
        
    Returns:
        1 for positive/neutral sentiment, 0 for negative
    """
    analyzer = get_sentiment_analyzer()
    score = analyzer.polarity_scores(text)['compound']
    return 1 if score >= 0 else 0


def load_dataset(use_cache=True):
    """
    Load the CMU-MOSI dataset with caching support to avoid reprocessing.
    
    Args:
        use_cache: Whether to use cached data if available
        
    Returns:
        Tuple of (train_data, test_data) where each is a list of dictionaries
        containing 'text', 'video_features', 'label', and 'video_id'
    """
    cache_file = os.path.join(CACHE_DIR, "preprocessed_data.npz")

    # Try to load from cache
    if use_cache and os.path.exists(cache_file):
        print("Loading preprocessed data from cache...")
        data = np.load(cache_file, allow_pickle=True)
        return data['train_data'].tolist(), data['test_data'].tolist()

    print("Processing dataset from scratch...")
    video_files = glob.glob(os.path.join(VIDEO_PATH, "*.mp4"))
    data = []

    for video_file in tqdm(video_files, desc="Processing Videos"):
        base_name = os.path.basename(video_file).split('.')[0]
        transcript_file = os.path.join(TRANSCRIPT_PATH, f"{base_name}.txt")

        if os.path.exists(transcript_file):
            # Read transcript
            with open(transcript_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            # Extract frames from video
            video_features = extract_video_frames(video_file, MAX_FRAMES)

            # Extract sentiment label
            label = extract_sentiment_label(text)

            data.append({
                'text': text,
                'video_features': video_features,
                'label': label,
                'video_id': base_name
            })

    # Visualize first video for debugging
    if video_files:
        print("Visualizing 5 frames of first video...")
        visualize_video_frames(video_files[0])

    # Split data into train and test sets
    train_data, test_data = train_test_split(
        data, 
        test_size=TEST_SPLIT, 
        random_state=RANDOM_SEED
    )

    # Save to cache
    if use_cache:
        print("Saving preprocessed data to cache...")
        np.savez(
            cache_file,
            train_data=np.array(train_data, dtype=object),
            test_data=np.array(test_data, dtype=object)
        )

    return train_data, test_data


def prepare_model_inputs(data, bert_tokenizer, gpt2_tokenizer, glove_tokenizer):
    """
    Prepare input data for all three models.
    
    This function uses pre-fitted tokenizers to avoid data leakage.
    The glove_tokenizer should be fit ONLY on training data.
    
    Args:
        data: List of data dictionaries
        bert_tokenizer: BERT tokenizer instance
        gpt2_tokenizer: GPT-2 tokenizer instance
        glove_tokenizer: Fitted Keras Tokenizer (fit on training data only!)
        
    Returns:
        Dictionary with processed inputs for each model
    """
    from text_utils import tokenize_for_bert, tokenize_for_gpt2, tokenize_for_glove
    
    # Extract texts, videos, and labels
    texts = [item['text'] for item in data]
    videos = np.array([item['video_features'] for item in data])
    labels = np.array([item['label'] for item in data])

    # Tokenize for each model
    text_input_bert, text_attention = tokenize_for_bert(texts, bert_tokenizer)
    text_input_gpt2 = tokenize_for_gpt2(texts, gpt2_tokenizer)
    text_input_glove = tokenize_for_glove(texts, glove_tokenizer)

    return {
        'text_input_bert': text_input_bert,
        'text_attention': text_attention,
        'text_input_gpt2': text_input_gpt2,
        'text_input_glove': text_input_glove,
        'video_input': videos,
        'labels': labels
    }
