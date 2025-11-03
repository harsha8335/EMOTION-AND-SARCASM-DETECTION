# EMOTION-AND-SARCASM-DETECTION
An NLP-based machine learning project that detects emotions and sarcasm in text, enabling emotionally intelligent AI communication.

💬 EMOTION AND SARCASM DETECTION SYSTEM
🎯 Objective

The primary objective of this project is to develop a machine learning-based system capable of detecting emotions and sarcasm in textual data.
The system combines Natural Language Processing (NLP) and Machine Learning (ML) techniques to accurately identify emotional states such as joy, anger, sadness, fear, disgust, shame, guilt and recognize sarcastic expressions where literal meanings contradict intended sentiments.

This includes:

Preprocessing raw text to remove noise and normalize input.

Extracting features using n-grams, TF-IDF, and word embeddings.

Training and evaluating ML models (LinearSVC & Logistic Regression).

Combining emotion and sarcasm detection to achieve more reliable sentiment interpretation.

Testing and validating model accuracy and efficiency.

The ultimate goal is to enable emotionally intelligent systems that understand human feelings and sarcasm for real-world applications like chatbots, customer feedback analysis, and social media monitoring.

📑 Table of Contents

Introduction & Objective

Literature Review

Proposed System

System Design

Software Description

Program Design

Testing

Conclusion

References

Appendix (Source Code, Sample Outputs, Screenshots)

📘 Project Overview

In today’s digital era, people frequently share emotions and opinions online through platforms like Twitter, Facebook, and chat applications.
Understanding these emotions accurately is crucial for customer feedback systems, mental health monitoring, and social media analytics.

Traditional sentiment analysis often fails to capture sarcasm, where the literal meaning differs from the intended emotion (e.g., “Oh great, my phone died again!”).

This project overcomes these challenges by developing an Emotion and Sarcasm Detection System using Python.
It employs Natural Language Processing (NLP) and Machine Learning models to classify text into emotional categories and detect sarcastic remarks.
The system learns linguistic and contextual cues from labeled datasets to predict both emotion and sarcasm effectively.

⚙️ Features

✅ Dual-Module Design: Emotion Detection + Sarcasm Detection
✅ Uses NLP preprocessing (tokenization, stopword removal, normalization)
✅ Employs LinearSVC and Logistic Regression for classification
✅ Handles sarcasm through rule-based and ML-based detection
✅ Supports real-time user input for prediction
✅ Outputs both emotion category and sarcasm status
✅ Achieved ~85% accuracy in emotion detection and ~80% accuracy in sarcasm detection

🧩 Tools & Technologies Used
Category	Tools / Technologies
Programming Language	Python 3.x
Libraries	scikit-learn, pandas, numpy, re, joblib, collections
Algorithms	Linear Support Vector Classifier (SVC), Logistic Regression
Feature Extraction	n-grams, TF-IDF
IDEs	Google Colab / VS Code / PyCharm
Dataset	Emotion & Sarcasm labeled datasets
Output Interface	Command-line console
💾 Dataset Description

Emotion Dataset: Contains labeled sentences representing multiple emotional states – joy, fear, anger, sadness, disgust, shame, guilt.

Sarcasm Dataset: Labeled comments marked as “Sarcastic” or “Not Sarcastic.”

Data is cleaned, tokenized, and vectorized using n-gram and TF-IDF features for model training.

🧠 Proposed System

The system processes raw text, removes noise, and transforms it into numerical vectors.
It consists of two main components:

1️⃣ Emotion Detection Module

Uses n-gram features and Linear Support Vector Classifier (SVC).

Classifies sentences into one or more emotional categories.

2️⃣ Sarcasm Detection Module

Uses TF-IDF features with Logistic Regression.

Incorporates rule-based heuristics (e.g., phrases like “yeah right”, “oh great”).

Detects sarcasm even in indirect or implicit cases.

🧩 System Design
🔹 Data Flow

User Input → User enters a sentence.

Preprocessing → Cleans text (removes punctuation, symbols, expands contractions).

Feature Extraction → Converts text into vectorized format (n-grams / TF-IDF).

Model Prediction →

Emotion model predicts the emotional state.

Sarcasm model predicts sarcastic intent.

Output Display → System displays both detected emotion and sarcasm status.

🧱 Architecture

Input Layer: Text sentence

Processing Layer: NLP preprocessing & feature extraction

Model Layer: Trained SVC and Logistic Regression models

Output Layer: Emotion + Sarcasm prediction

💻 Program Design
🧩 Emotion Detection Module

Model: LinearSVC (Support Vector Machine)

Feature Extraction: n-grams + DictVectorizer

Dataset Split: 70% training / 30% testing

Output Example:

Input: “Everything feels perfect today”  
Output: Emotion → Joy

🧩 Sarcasm Detection Module

Model: Logistic Regression

Feature Extraction: TF-IDF (1–2 grams)

Dataset Split: 80% training / 20% testing

Output Example:

Input: “Oh wonderful, it’s raining again on my day off.”  
Output: Sarcastic

🧪 Model Training & Evaluation
Module	Algorithm	Feature Extraction	Accuracy
Emotion Detection	Linear SVC	n-gram + CountVectorizer	~85%
Sarcasm Detection	Logistic Regression	TF-IDF (1–2 grams)	~80%

Evaluation Metrics:

Precision: 0.84 (Emotion), 0.79 (Sarcasm)

Recall: 0.83 (Emotion), 0.80 (Sarcasm)

F1-Score: 0.84 (Emotion), 0.79 (Sarcasm)

🔬 Testing
✅ Test Plan

Validate emotion classification accuracy.

Verify sarcasm detection functionality.

Check for handling of invalid or empty inputs.

Confirm integration between both models.

✅ Test Results
Input	Detected Emotion	Sarcasm
“Everything feels perfect today.”	Joy	Not Sarcastic
“Oh great, another meeting at 7 a.m.!”	Disgust	Sarcastic
“I can’t even look at myself after what I did.”	Shame	Not Sarcastic
📊 Performance Summary

Emotion Detection Model

Training Accuracy: ~90%

Testing Accuracy: ~85%

Sarcasm Detection Model

Training Accuracy: ~87%

Testing Accuracy: ~80%

Both models show strong generalization and minimal overfitting.
TF-IDF features captured phrase-level sarcasm effectively, while n-grams captured emotional tone.

💡 Applications

✅ Customer Feedback Analysis
✅ Social Media Sentiment Tracking
✅ Chatbots & Virtual Assistants
✅ Mental Health Text Monitoring
✅ Emotionally Intelligent AI Systems

🚀 Insights & Future Scope

Future improvements include using deep learning (LSTM, BERT) for better contextual understanding.

Multilingual support can enhance usability across languages.

Integration into real-time chatbots or social media dashboards could provide live emotion & sarcasm analysis.

✅ Conclusion

This project successfully demonstrates how machine learning and NLP can be combined to detect both emotions and sarcasm in textual data.
By integrating these two aspects, the system achieves a deeper understanding of human communication, enabling applications in marketing, social analytics, and human-computer interaction.

The dual-model framework provides a reliable approach for emotion classification and sarcasm detection, paving the way for emotionally aware AI systems capable of understanding nuanced human language.

📚 References

Investigations in Computational Sarcasm — Aditya Joshi et al., 2018

Emotion Detection in NLP — Federica Cavicchio, 2024

Sentiment Analysis: Mining Opinions and Emotions — Bing Liu, 2015

Deep Learning Approaches for Sentiment Analysis — Basant Agarwal et al., 2020

Computational Intelligence Methods for Sentiment Analysis — D. Jude Hemanth, 2024


