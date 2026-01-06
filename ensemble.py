"""
Stacking Ensemble for the Multimodal Sentiment Analysis project.

Contains the meta-learner that combines predictions from IFFSA, BFSA, and TBJE.
Uses Logistic Regression as the meta-learner.
"""

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

from config import ENSEMBLE_MODEL_PATH, RANDOM_SEED


def build_ensemble_model(iffsa_model, bfsa_model, tbje_model, val_data):
    """
    Build a stacking ensemble model using logistic regression as meta-learner.
    
    The meta-learner is trained on validation data predictions, using:
    - Average of IFFSA and BFSA predictions
    - TBJE predictions
    
    Args:
        iffsa_model: Trained IFFSA model
        bfsa_model: Trained BFSA model
        tbje_model: Trained TBJE model
        val_data: Validation data dictionary (held-out from training!)
        
    Returns:
        Trained LogisticRegression meta-learner
    """
    print("Getting predictions from base models for ensemble training...")
    
    # Get predictions from each base model
    iffsa_preds = iffsa_model.predict([
        val_data['text_input_bert'],
        val_data['text_attention'],
        val_data['video_input']
    ], verbose=0)

    bfsa_preds = bfsa_model.predict([
        val_data['text_input_gpt2'],
        val_data['video_input']
    ], verbose=0)

    tbje_preds = tbje_model.predict([
        val_data['text_input_glove'],
        val_data['video_input']
    ], verbose=0)

    # Prepare features for meta-learner
    # Average IFFSA and BFSA predictions
    avg_iffsa_bfsa = (iffsa_preds + bfsa_preds) / 2

    # Create feature matrix: [avg_iffsa_bfsa, tbje_preds]
    X = np.concatenate([avg_iffsa_bfsa, tbje_preds], axis=1)
    Y = val_data['labels']

    # Train meta-learner
    meta_learner = LogisticRegression(random_state=RANDOM_SEED)
    meta_learner.fit(X, Y)
    
    print("Ensemble meta-learner trained successfully")
    return meta_learner


def get_ensemble_predictions(iffsa_model, bfsa_model, tbje_model, meta_learner, data):
    """
    Get ensemble predictions for new data.
    
    Args:
        iffsa_model: Trained IFFSA model
        bfsa_model: Trained BFSA model
        tbje_model: Trained TBJE model
        meta_learner: Trained meta-learner
        data: Data dictionary with model inputs
        
    Returns:
        Binary predictions from the ensemble
    """
    # Get base model predictions
    iffsa_preds = iffsa_model.predict([
        data['text_input_bert'],
        data['text_attention'],
        data['video_input']
    ], verbose=0)

    bfsa_preds = bfsa_model.predict([
        data['text_input_gpt2'],
        data['video_input']
    ], verbose=0)

    tbje_preds = tbje_model.predict([
        data['text_input_glove'],
        data['video_input']
    ], verbose=0)

    # Prepare features for meta-learner
    avg_iffsa_bfsa = (iffsa_preds + bfsa_preds) / 2
    X = np.concatenate([avg_iffsa_bfsa, tbje_preds], axis=1)

    # Get ensemble predictions
    return meta_learner.predict(X), iffsa_preds, bfsa_preds, tbje_preds


def save_ensemble(meta_learner, path=ENSEMBLE_MODEL_PATH):
    """
    Save the meta-learner to disk using joblib.
    
    Note: sklearn models should use joblib, NOT Keras .save()
    
    Args:
        meta_learner: Trained LogisticRegression model
        path: File path to save to
    """
    joblib.dump(meta_learner, path)
    print(f"Ensemble model saved to {path}")


def load_ensemble(path=ENSEMBLE_MODEL_PATH):
    """
    Load the meta-learner from disk.
    
    Args:
        path: File path to load from
        
    Returns:
        Loaded LogisticRegression model
    """
    meta_learner = joblib.load(path)
    print(f"Ensemble model loaded from {path}")
    return meta_learner
