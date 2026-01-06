"""
Training and evaluation pipeline for the Multimodal Sentiment Analysis project.

Contains the main training loop, model evaluation, and orchestration logic.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from config import (
    EPOCHS, BATCH_SIZE, VALIDATION_SPLIT, RANDOM_SEED,
    IFFSA_MODEL_PATH, BFSA_MODEL_PATH, TBJE_MODEL_PATH
)
from data import load_dataset, prepare_model_inputs, clear_memory
from text_utils import (
    create_bert_tokenizer, create_gpt2_tokenizer, create_glove_tokenizer,
    save_glove_tokenizer
)
from models import build_iffsa_model, build_bfsa_model, build_tbje_model
from ensemble import build_ensemble_model, get_ensemble_predictions, save_ensemble


def create_tokenizers(train_texts):
    """
    Create all tokenizers needed for the models.
    
    IMPORTANT: The GloVe tokenizer is fit ONLY on training texts to prevent
    data leakage. The same tokenizer is then used for validation and test data.
    
    Args:
        train_texts: List of training text strings
        
    Returns:
        Tuple of (bert_tokenizer, gpt2_tokenizer, glove_tokenizer)
    """
    print("Creating tokenizers...")
    
    # BERT tokenizer (pretrained, no fitting needed)
    bert_tokenizer = create_bert_tokenizer()
    
    # GPT-2 tokenizer (pretrained, no fitting needed)
    gpt2_tokenizer = create_gpt2_tokenizer()
    
    # Keras tokenizer for GloVe (fit on training data only!)
    glove_tokenizer = create_glove_tokenizer(train_texts)
    
    print("Tokenizers created successfully")
    return bert_tokenizer, gpt2_tokenizer, glove_tokenizer


def split_for_ensemble(processed_data, ensemble_val_ratio=0.3):
    """
    Split training data to create a held-out validation set for ensemble training.
    
    This is critical to avoid data leakage: the meta-learner should be trained
    on data that the base models have NOT seen during training.
    
    Args:
        processed_data: Dictionary with processed model inputs
        ensemble_val_ratio: Fraction to hold out for ensemble training
        
    Returns:
        Tuple of (train_data, ensemble_val_data) dictionaries
    """
    n_samples = len(processed_data['labels'])
    indices = np.arange(n_samples)
    
    train_idx, val_idx = train_test_split(
        indices,
        test_size=ensemble_val_ratio,
        random_state=RANDOM_SEED
    )
    
    # Helper to split a dictionary
    def split_dict(data, indices):
        result = {}
        for key, value in data.items():
            if hasattr(value, 'numpy'):
                # TensorFlow tensor
                result[key] = tf.gather(value, indices)
            else:
                # Numpy array
                result[key] = value[indices]
        return result
    
    return split_dict(processed_data, train_idx), split_dict(processed_data, val_idx)


def train_single_model(model, inputs, labels, model_name):
    """
    Train a single model with early stopping.
    
    Args:
        model: Keras model to train
        inputs: List of input tensors
        labels: Target labels
        model_name: Name for logging
        
    Returns:
        Trained model
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True,
            monitor='val_loss',
            verbose=1
        )
    ]
    
    model.fit(
        inputs,
        labels,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    return model


def evaluate_single_model(model, inputs, labels, model_name):
    """
    Evaluate a single model and return metrics.
    
    Args:
        model: Trained Keras model
        inputs: List of input tensors
        labels: True labels
        model_name: Name for logging
        
    Returns:
        Tuple of (predictions, accuracy, f1_score)
    """
    preds = model.predict(inputs, verbose=0)
    binary_preds = (preds > 0.5).astype(int).flatten()
    
    accuracy = accuracy_score(labels, binary_preds)
    f1 = f1_score(labels, binary_preds)
    
    print(f"{model_name}: Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}")
    
    return preds, accuracy, f1


def train_and_evaluate():
    """
    Main training and evaluation pipeline.
    
    This function:
    1. Loads the dataset
    2. Creates tokenizers (fitting GloVe on training data only)
    3. Preprocesses data for all models
    4. Splits data for proper ensemble training (avoiding data leakage)
    5. Trains TBJE, IFFSA, and BFSA models
    6. Trains the stacking ensemble meta-learner
    7. Evaluates all models and prints results
    """
    # Step 1: Load dataset
    print("\n" + "="*60)
    print("Loading dataset...")
    print("="*60)
    train_data, test_data = load_dataset()
    print(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    # Step 2: Create tokenizers
    # IMPORTANT: Fit GloVe tokenizer ONLY on training texts
    train_texts = [item['text'] for item in train_data]
    bert_tokenizer, gpt2_tokenizer, glove_tokenizer = create_tokenizers(train_texts)
    
    # Save the GloVe tokenizer for later use
    save_glove_tokenizer(glove_tokenizer)
    
    # Step 3: Preprocess data
    print("\nProcessing training data...")
    processed_train_data = prepare_model_inputs(
        train_data, bert_tokenizer, gpt2_tokenizer, glove_tokenizer
    )
    
    print("Processing test data...")
    processed_test_data = prepare_model_inputs(
        test_data, bert_tokenizer, gpt2_tokenizer, glove_tokenizer
    )
    
    # Step 4: Split training data for ensemble
    # The base models will be trained on train_for_base
    # The meta-learner will be trained on ensemble_val_data (held-out)
    train_for_base, ensemble_val_data = split_for_ensemble(processed_train_data)
    
    print(f"\nBase model training samples: {len(train_for_base['labels'])}")
    print(f"Ensemble validation samples: {len(ensemble_val_data['labels'])}")
    
    # Step 5: Train base models
    # Train TBJE (lightest model, train first)
    tbje_model = build_tbje_model()
    tbje_model = train_single_model(
        tbje_model,
        [train_for_base['text_input_glove'], train_for_base['video_input']],
        train_for_base['labels'],
        "TBJE"
    )
    tbje_model.save(TBJE_MODEL_PATH)
    print(f"TBJE model saved to {TBJE_MODEL_PATH}")
    
    # Clear memory before next model
    clear_memory()
    
    # Train IFFSA (BERT-based)
    iffsa_model = build_iffsa_model()
    iffsa_model = train_single_model(
        iffsa_model,
        [train_for_base['text_input_bert'], 
         train_for_base['text_attention'], 
         train_for_base['video_input']],
        train_for_base['labels'],
        "IFFSA"
    )
    iffsa_model.save(IFFSA_MODEL_PATH)
    print(f"IFFSA model saved to {IFFSA_MODEL_PATH}")
    
    clear_memory()
    
    # Train BFSA (GPT-2 based)
    bfsa_model = build_bfsa_model()
    bfsa_model = train_single_model(
        bfsa_model,
        [train_for_base['text_input_gpt2'], train_for_base['video_input']],
        train_for_base['labels'],
        "BFSA"
    )
    bfsa_model.save(BFSA_MODEL_PATH)
    print(f"BFSA model saved to {BFSA_MODEL_PATH}")
    
    # Step 6: Train ensemble meta-learner on held-out validation data
    print("\n" + "="*60)
    print("Training ensemble meta-learner...")
    print("="*60)
    meta_learner = build_ensemble_model(
        iffsa_model, bfsa_model, tbje_model, ensemble_val_data
    )
    save_ensemble(meta_learner)
    
    # Step 7: Evaluate all models on test set
    print("\n" + "="*60)
    print("Evaluating models on test set...")
    print("="*60)
    
    # Evaluate IFFSA
    iffsa_preds, iffsa_acc, iffsa_f1 = evaluate_single_model(
        iffsa_model,
        [processed_test_data['text_input_bert'],
         processed_test_data['text_attention'],
         processed_test_data['video_input']],
        processed_test_data['labels'],
        "IFFSA"
    )
    
    # Evaluate BFSA
    bfsa_preds, bfsa_acc, bfsa_f1 = evaluate_single_model(
        bfsa_model,
        [processed_test_data['text_input_gpt2'],
         processed_test_data['video_input']],
        processed_test_data['labels'],
        "BFSA"
    )
    
    # Evaluate TBJE
    tbje_preds, tbje_acc, tbje_f1 = evaluate_single_model(
        tbje_model,
        [processed_test_data['text_input_glove'],
         processed_test_data['video_input']],
        processed_test_data['labels'],
        "TBJE"
    )
    
    # Evaluate Ensemble
    ensemble_preds, _, _, _ = get_ensemble_predictions(
        iffsa_model, bfsa_model, tbje_model, meta_learner, processed_test_data
    )
    ensemble_acc = accuracy_score(processed_test_data['labels'], ensemble_preds)
    ensemble_f1 = f1_score(processed_test_data['labels'], ensemble_preds)
    print(f"Ensemble: Accuracy = {ensemble_acc:.4f}, F1 Score = {ensemble_f1:.4f}")
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"{'Model':<12} {'Accuracy':>10} {'F1 Score':>10}")
    print("-"*34)
    print(f"{'IFFSA':<12} {iffsa_acc:>10.4f} {iffsa_f1:>10.4f}")
    print(f"{'BFSA':<12} {bfsa_acc:>10.4f} {bfsa_f1:>10.4f}")
    print(f"{'TBJE':<12} {tbje_acc:>10.4f} {tbje_f1:>10.4f}")
    print(f"{'Ensemble':<12} {ensemble_acc:>10.4f} {ensemble_f1:>10.4f}")
    print("="*60)
    
    return {
        'iffsa': {'accuracy': iffsa_acc, 'f1': iffsa_f1},
        'bfsa': {'accuracy': bfsa_acc, 'f1': bfsa_f1},
        'tbje': {'accuracy': tbje_acc, 'f1': tbje_f1},
        'ensemble': {'accuracy': ensemble_acc, 'f1': ensemble_f1}
    }
