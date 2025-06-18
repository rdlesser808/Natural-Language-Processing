# Natural-Language-Processing-with-Deep-Learning
Master's level coursework from Natural Language Processing with Deep Learning module featuring comprehensive neural network implementation, comparative model analysis (CNN vs RNN-LSTM), advanced text preprocessing pipelines, and deep learning optimization techniques. Demonstrates technical proficiency in NLP while addressing critical ethical issues including algorithmic bias, content moderation transparency, and responsible AI deployment in information integrity contexts.

# Fake News Detection: CNN vs RNN-LSTM
This repository contains research and implementation of neural network models for detecting fake news using natural language processing and deep learning techniques. The work includes comparative analysis of Convolutional Neural Networks and Recurrent Neural Networks with their effectiveness in identifying misinformation.

## Overview
Fake news has become a pervasive issue, significantly impacting public perception, political stability, and social harmony. This project aims to develop accurate, scalable detection systems using deep learning to assist in preserving information integrity and safeguarding democratic processes.

## Key Objectives
- Develop neural network models for fake news classification
- Compare effectiveness of CNN vs RNN-LSTM architectures
- Create robust text preprocessing pipelines for news content
- Address ethical considerations in ML-based content moderation

## Dataset
### Primary Dataset - Fake News Detection Dataset
- **Size**: 44,898 samples, 4 features, 2 classes
- **Source**: Public domain via Kaggle
- **Format**: CSV (Comma Separated Values)
- **Distribution**: Mixed real and fake news articles
- **Source**: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

### Testing Dataset - UTK Machine Learning Club
- **Size**: 5,200 samples, 2 features, 2 classes
- **Source**: Kaggle competition dataset
- **Purpose**: External validation and generalization testing
- **Source**: https://kaggle.com/competitions/fake-news

## Features
The dataset includes news article components:
- **Title**: Article headlines
- **Text**: Full article content
- **Subject**: Topic categories
- **Date**: Publication timestamps

## Target Classes
- **Real (0)**: Legitimate news from Reuters.com
- **Fake (1)**: Misinformation from unreliable sources flagged by Politifact and Wikipedia

## Methodology
### Models Implemented

**Convolutional Neural Network (CNN)**
- Excellent at detecting local patterns in text embeddings
- 1D convolutions with max pooling for feature extraction
- Spatial hierarchy recognition in textual data
- Superior performance in identifying fake news patterns

**Recurrent Neural Network with LSTM**
- Captures sequential dependencies and context
- Long Short-Term Memory for temporal relationships
- Processes text as sequential data
- Better balanced performance across classes

### Model Architectures

**CNN Architecture**
```
- Embedding Layer (100-dimensional vectors)
- 1D Convolutional Layer (64 filters, kernel size 5)
- Max Pooling Layer (pool size 2)  
- Dropout Layer (0.3 rate)
- Dense Layer (64 units, ReLU)
- Output Layer (1 unit, sigmoid)
```
**Parameters**: 19,078,429 total

**RNN-LSTM Architecture**
```
- Embedding Layer (100-dimensional vectors)
- LSTM Layer (64 units)
- Dropout Layer (regularization)
- Dense Layer (64 units, ReLU)
- Output Layer (1 unit, sigmoid)
```
**Parameters**: 17,953,765 total

## Model Evaluation Metrics
- **Accuracy**: Overall classification effectiveness
- **Precision**: Agreement of data labels with positive predictions
- **Recall**: Model's ability to identify class labels
- **F1-Score**: Harmonic mean of precision and recall

## Results Summary
### Final Model Performance (External Test Dataset)

| Model | Accuracy | Precision (Real) | Precision (Fake) | Recall (Real) | Recall (Fake) | F1 (Real) | F1 (Fake) |
|-------|----------|------------------|------------------|---------------|---------------|-----------|-----------|
| CNN | 0.60 | 0.67 | 0.59 | 0.21 | 0.91 | 0.33 | 0.72 |
| RNN-LSTM | 0.57 | 0.53 | 0.59 | 0.42 | 0.69 | 0.47 | 0.64 |

### Key Findings
- **CNN Strengths**: Higher overall accuracy (60% vs 57%), excellent fake news detection (91% recall)
- **RNN-LSTM Strengths**: More balanced performance, better real news identification (42% vs 21% recall)
- **Overfitting Challenge**: Both models achieved 99-100% accuracy on training data but lower external performance
- **Generalization Gap**: Significant performance drop on unseen dataset indicates overfitting

## Technical Implementation
### Software & Libraries
- **Python 3**: Primary programming language
- **TensorFlow/Keras**: Deep learning framework
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **scikit-learn**: Model evaluation and preprocessing
- **NLTK**: Natural language processing toolkit
- **matplotlib**: Data visualization
- **Jupyter Notebooks**: Development environment

### Text Preprocessing Pipeline
- **Data Cleaning**: Remove duplicates and invalid entries
- **Text Normalization**: Lowercase conversion, URL removal
- **Contraction Expansion**: Standardize abbreviated forms
- **Stopword Removal**: Eliminate common words using NLTK
- **Lemmatization**: Reduce words to root forms
- **Tokenization**: Convert text to integer sequences
- **Padding**: Uniform sequence lengths (95th percentile)

## Repository Structure
```
├── NLP AE1/
│   └── NLP AE1.ipynb                # Assignment 1 implementation
├── NLP AE2/
│   ├── NLP AE2.pdf                  # Complete research paper and methodology
│   ├── NLP AE2 CNN Final.ipynb      # Final CNN implementation
│   ├── NLP AE2 RNN Final.ipynb      # Final RNN-LSTM implementation
│   └── NLP AE2 CNN with dates.ipynb # Original CNN with temporal features
└── README.md                        # Project documentation
```

## Limitations & Future Work
### Current Limitations
- **Small External Dataset**: Limited generalization testing
- **Overfitting Issues**: High training accuracy doesn't translate to real-world performance
- **Computing Constraints**: Limited hyperparameter tuning capabilities
- **Dataset Bias**: Potential bias toward specific news sources and topics

### Future Improvements
- **Larger Datasets**: Expand training data with diverse news sources
- **Advanced Architectures**: Implement transformer models (BERT, RoBERTa)
- **Hybrid Models**: Combine CNN and LSTM advantages
- **Feature Engineering**: Incorporate source credibility and metadata
- **Cloud Computing**: Utilize distributed training for extensive optimization
- **Cross-validation**: Implement robust validation strategies

## Ethical Considerations
### Algorithmic Bias
- **Source Discrimination**: Risk of unfairly targeting non-mainstream outlets
- **Content Bias**: Potential bias against certain political perspectives
- **Historical Bias**: Training data may reflect existing prejudices

### Transparency & Accountability
- **Black Box Problem**: Deep learning models lack interpretability
- **Decision Transparency**: Need for explainable AI in content moderation
- **Accountability Framework**: Clear responsibility for automated decisions

### Privacy & Data Protection
- **Data Access**: Extensive text data processing requirements
- **User Privacy**: Protection of personal information in news consumption
- **Consent Protocols**: Transparent data usage policies

### Societal Impact
- **Over-reliance Risk**: Automated systems may suppress legitimate discourse
- **False Security**: Imperfect detection may create overconfidence
- **Democratic Processes**: Impact on free speech and information flow

## Research Papers
This repository includes comprehensive research documentation:
- **Complete Methodology**: Detailed experimental procedures and model architectures
- **Comparative Analysis**: In-depth evaluation of CNN vs RNN-LSTM performance
- **Ethical Framework**: Thorough discussion of responsible AI deployment
- **Future Directions**: Recommendations for continued research and development

## Acknowledgments
- **Kaggle Community**: Dataset availability and competition platform
- **Reuters**: Reliable news source for training data
- **Politifact & Wikipedia**: Fake news identification and labeling
- **Research Community**: Natural language processing and deep learning methodologies

**Disclaimer**: This research is for academic purposes only. The models are not intended for production content moderation without extensive validation and human oversight. Always consider multiple sources and professional fact-checking for critical information verification.
