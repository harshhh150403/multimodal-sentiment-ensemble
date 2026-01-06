"""
Model definitions for the Multimodal Sentiment Analysis project.

Contains the three base models:
- IFFSA: BERT + ResNet50 with Multi-Head Attention fusion
- BFSA: GPT-2 + VGG16 with Bilinear fusion
- TBJE: GloVe/LSTM + CNN with Transformer fusion
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Input, MultiHeadAttention, Concatenate,
    LSTM, Bidirectional, GlobalAveragePooling1D, Flatten,
    TimeDistributed, Conv2D, MaxPooling2D, Embedding,
    LayerNormalization, Add, Lambda
)
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.optimizers import Adam
from transformers import TFBertModel, TFGPT2Model

from config import (
    MAX_FRAMES, MAX_TEXT_LENGTH, LEARNING_RATE,
    VOCAB_SIZE, EMBEDDING_DIM, FRAME_SIZE
)


# ==============================================================================
# CUSTOM KERAS LAYERS
# Defined at module level for reusability and proper serialization
# ==============================================================================

class BertLayer(tf.keras.layers.Layer):
    """Custom Keras layer wrapper for BERT model."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')
    
    def call(self, inputs):
        input_ids, attention_mask = inputs
        return self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]


class GPT2Layer(tf.keras.layers.Layer):
    """Custom Keras layer wrapper for GPT-2 model."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gpt2 = TFGPT2Model.from_pretrained('gpt2')
    
    def call(self, input_ids):
        return self.gpt2(input_ids=input_ids)[0]


class AttentionPrepLayer(tf.keras.layers.Layer):
    """Reshape features for multi-head attention by adding sequence dimension."""
    
    def call(self, features):
        return tf.expand_dims(features, axis=1)


class BilinearInteraction(tf.keras.layers.Layer):
    """Compute element-wise multiplication for bilinear fusion."""
    
    def call(self, inputs):
        text, video = inputs
        return tf.keras.layers.multiply([text, video])


# ==============================================================================
# MODEL BUILDERS
# ==============================================================================

def build_iffsa_model():
    """
    Build the IFFSA (Interactive Feature Fusion for Sentiment Analysis) model.
    
    Architecture:
    - Text: BERT → Dense(512)
    - Video: ResNet50 (frozen) → GlobalAvgPool → Dense(512)
    - Fusion: Multi-Head Attention + Concatenation
    - Output: Dense(256) → Dropout → Sigmoid
    
    Returns:
        Compiled Keras Model
    """
    # Input layers
    text_input = Input(shape=(MAX_TEXT_LENGTH,), dtype=tf.int32, name='text_input')
    text_attention = Input(shape=(MAX_TEXT_LENGTH,), dtype=tf.int32, name='text_attention')
    video_input = Input(shape=(MAX_FRAMES, FRAME_SIZE, FRAME_SIZE, 3), name='video_input')

    # Text processing with BERT
    bert_outputs = BertLayer()([text_input, text_attention])
    text_features = GlobalAveragePooling1D()(bert_outputs)
    text_features = Dense(512, activation='relu')(text_features)

    # Video processing with ResNet50
    resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    resnet.trainable = False  # Freeze pretrained weights
    
    video_processor = TimeDistributed(resnet)(video_input)
    video_features = GlobalAveragePooling1D()(video_processor)
    video_features = Dense(512, activation='relu')(video_features)

    # Attention-based fusion
    text_for_attention = AttentionPrepLayer()(text_features)
    video_for_attention = AttentionPrepLayer()(video_features)

    attn_output = MultiHeadAttention(num_heads=8, key_dim=64)(
        query=text_for_attention,
        value=video_for_attention
    )
    attn_output = Flatten()(attn_output)

    # Final fusion and classification
    fused = Concatenate()([text_features, video_features, attn_output])
    fused = Dense(256, activation='relu')(fused)
    fused = Dropout(0.5)(fused)
    output = Dense(1, activation='sigmoid')(fused)

    model = Model(
        inputs=[text_input, text_attention, video_input],
        outputs=output,
        name='IFFSA'
    )
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_bfsa_model():
    """
    Build the BFSA (Bilinear Fusion for Sentiment Analysis) model.
    
    Architecture:
    - Text: GPT-2 → Dense(512)
    - Video: VGG16 (frozen) → GlobalAvgPool → Dense(512)
    - Fusion: Bilinear (element-wise multiply) + Concatenation
    - Output: Dense(256) → Dropout → Sigmoid
    
    Returns:
        Compiled Keras Model
    """
    # Input layers
    text_input = Input(shape=(MAX_TEXT_LENGTH,), dtype=tf.int32, name='text_input')
    video_input = Input(shape=(MAX_FRAMES, FRAME_SIZE, FRAME_SIZE, 3), name='video_input')

    # Text processing with GPT-2
    gpt2_outputs = GPT2Layer()(text_input)
    text_features = GlobalAveragePooling1D()(gpt2_outputs)
    text_features = Dense(512, activation='relu')(text_features)

    # Video processing with VGG16
    vgg = VGG16(weights='imagenet', include_top=False, pooling='avg')
    vgg.trainable = False  # Freeze pretrained weights
    
    video_features = TimeDistributed(vgg)(video_input)
    video_features = GlobalAveragePooling1D()(video_features)
    video_features = Dense(512, activation='relu')(video_features)

    # Bilinear fusion
    bilinear = BilinearInteraction()([text_features, video_features])

    # Final fusion and classification
    fused = Concatenate()([text_features, video_features, bilinear])
    fused = Dense(256, activation='relu')(fused)
    fused = Dropout(0.5)(fused)
    output = Dense(1, activation='sigmoid')(fused)

    model = Model(
        inputs=[text_input, video_input],
        outputs=output,
        name='BFSA'
    )
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_tbje_model():
    """
    Build the TBJE (Text-Based Joint Embedding) model.
    
    Architecture:
    - Text: Embedding → Bi-LSTM → GlobalAvgPool
    - Video: CNN (Conv2D×2) → Dense(128) reduction → LSTM
    - Fusion: Transformer (Multi-Head Attention + Residual + LayerNorm)
    - Output: Dense(256) → Dropout → Sigmoid
    
    This is a lighter model compared to IFFSA and BFSA, using simpler
    architectures instead of heavy pretrained models.
    
    Returns:
        Compiled Keras Model
    """
    # Text input branch
    text_input = Input(shape=(MAX_TEXT_LENGTH,), dtype=tf.int32, name='text_input')
    
    # Embedding layer (could load pretrained GloVe weights here)
    embedding_layer = Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_TEXT_LENGTH
    )
    text_embeddings = embedding_layer(text_input)
    
    # Bidirectional LSTM for text
    text_lstm = Bidirectional(LSTM(128, return_sequences=True))(text_embeddings)
    text_features = GlobalAveragePooling1D()(text_lstm)

    # Video input branch
    video_input = Input(shape=(MAX_FRAMES, FRAME_SIZE, FRAME_SIZE, 3), name='video_input')

    # Lightweight CNN for frame feature extraction
    conv1 = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(video_input)
    pool1 = TimeDistributed(MaxPooling2D((2, 2)))(conv1)
    
    conv2 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(pool1)
    pool2 = TimeDistributed(MaxPooling2D((2, 2)))(conv2)

    # Flatten and reduce dimensionality before LSTM
    flattened = TimeDistributed(Flatten())(pool2)
    dense_reduce = TimeDistributed(Dense(128, activation='relu'))(flattened)

    # LSTM for temporal modeling
    video_lstm = LSTM(128, return_sequences=True)(dense_reduce)
    video_features = GlobalAveragePooling1D()(video_lstm)

    # Transformer fusion
    text_reshaped = Lambda(lambda x: tf.expand_dims(x, axis=1))(text_features)
    video_reshaped = Lambda(lambda x: tf.expand_dims(x, axis=1))(video_features)
    concat_features = Concatenate(axis=1)([text_reshaped, video_reshaped])

    # Self-attention with residual connection
    attn_output = MultiHeadAttention(num_heads=4, key_dim=64)(concat_features, concat_features)
    summed = Add()([concat_features, attn_output])
    transformer_output = LayerNormalization()(summed)

    # Final classification
    fused_features = Flatten()(transformer_output)
    fused_features = Dense(256, activation='relu')(fused_features)
    fused_features = Dropout(0.5)(fused_features)
    output = Dense(1, activation='sigmoid')(fused_features)

    model = Model(
        inputs=[text_input, video_input],
        outputs=output,
        name='TBJE'
    )
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model
