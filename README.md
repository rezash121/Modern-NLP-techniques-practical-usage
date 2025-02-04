# Modern-NLP-techniques-practical-usage

# Kaggle + Hugging Face Integration for Sentiment Analysis

This repository demonstrates how to integrate Kaggle with Hugging Face to build a sentiment analysis pipeline. The project covers:

- **Loading and Preprocessing a Kaggle Dataset:** Using the Sentiment140 dataset for sentiment analysis.
- **Training a Transformer Model:** Training a small Hugging Face Transformer model (e.g., DistilBERT) on Kaggle.
- **Saving the Model and Tokenizer:** Persisting the trained model and tokenizer.
- **Evaluating the Model on Google Colab:** Loading and evaluating the saved model in Google Colab.
- **(Optional) Visualizing Attention:** Visualizing attention weights of the model for deeper insights.


## Overview

This project shows how to:
1. Log in to Kaggle, load, and preprocess a sentiment analysis dataset (Sentiment140).
2. Train a small Transformer model (DistilBERT) using Hugging Face’s Transformers library.
3. Save the trained model and tokenizer.
4. Evaluate the model on Google Colab.
5. (Optional) Visualize the model’s attention to understand its focus.


## Prerequisites

- **Kaggle Account:** To access and train on the Sentiment140 dataset.
- **Google Colab Account:** For model evaluation.
- **Python 3.7+** and relevant libraries: `transformers`, `datasets`, `scikit-learn`, `torch`, `bertviz` (for attention visualization), etc.
- **Kaggle API Token:** (if needed) for dataset access.



