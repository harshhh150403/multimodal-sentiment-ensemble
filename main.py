"""
Main entry point for the Multimodal Sentiment Analysis project.

This script orchestrates the entire training and evaluation pipeline.
Run this file to train all models and evaluate performance.

Usage:
    python main.py
"""

from config import configure_gpu, set_seeds, create_directories
from train import train_and_evaluate


def main():
    """
    Main function to run the multimodal sentiment analysis pipeline.
    
    Steps:
    1. Configure GPU memory growth
    2. Set random seeds for reproducibility
    3. Create necessary directories
    4. Run training and evaluation
    """
    print("\n" + "="*60)
    print("MULTIMODAL SENTIMENT ANALYSIS")
    print("Ensemble of IFFSA, BFSA, and TBJE Models")
    print("="*60 + "\n")
    
    # Setup
    print("Setting up environment...")
    configure_gpu()
    set_seeds()
    create_directories()
    
    # Run training and evaluation
    print("\nStarting training pipeline...\n")
    results = train_and_evaluate()
    
    print("\nTraining complete!")
    return results


if __name__ == "__main__":
    main()
