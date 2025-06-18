# Natural-Language-Processing-with-Deep-Learning
Master's level coursework from Natural Language Processing with Deep Learning module featuring comprehensive neural network implementation, comparative model analysis, advanced text preprocessing pipelines, and deep learning optimization techniques. Demonstrates technical proficiency in NLP while addressing critical ethical issues including algorithmic bias, content moderation transparency, and responsible AI deployment in information integrity contexts.

## Projects Overview

### 1. Sentiment Analysis: Naïve Bayes vs BiLSTM
**Location:** `NLP AE1/`

A comparative study implementing traditional machine learning (Naïve Bayes with Bag-of-Words) against deep learning (Bidirectional LSTM with GloVe embeddings) for movie review sentiment classification.

#### Project Components
1. **Text Normalization**: Comprehensive preprocessing pipeline including lowercasing, punctuation removal, tokenization, and lemmatization
2. **Named Entity Recognition**: SpaCy-based NER extraction from movie review text
3. **Naïve Bayes Classifier**: Multinomial NB with BoW vectorization and feature selection (top 3000 features)
4. **BiLSTM Neural Network**: Bidirectional LSTM with pre-trained GloVe embeddings and regularization techniques

#### Dataset
- **Source**: IMDB Movie Reviews (40,000 samples)
- **Features**: Movie review text
- **Classes**: Positive vs Negative sentiment
- **Split**: 90% training, 10% testing

#### Model Performance
**Test Dataset Results:**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naïve Bayes | 0.84 | 0.84 | 0.84 | 0.84 |
| BiLSTM | 0.80 | 0.80 | 0.80 | 0.80 |

**Key Insights:**
- Naïve Bayes outperformed BiLSTM across all metrics
- Simpler models can be highly effective for straightforward classification tasks
- BiLSTM's regularization techniques (dropout, L2) may have led to underfitting
- Demonstrates importance of model selection based on task complexity

#### Technical Features
- **Text Processing**: NLTK for tokenization and lemmatization
- **NER**: SpaCy English language model for entity extraction
- **Feature Engineering**: Chi-square feature selection for dimensionality reduction
- **Deep Learning**: TensorFlow/Keras with early stopping and regularization
- **Embeddings**: Pre-trained GloVe 100-dimensional word vectors

### 2. Fake News Detection: CNN vs RNN-LSTM
**Location:** `NLP AE2/`

A comprehensive research project comparing Convolutional Neural Networks and Recurrent Neural Networks with Long Short-Term Memory for detecting fake news using natural language processing and deep learning techniques.

#### Key Objectives
- Develop neural network models for fake news classification
- Compare effectiveness of CNN vs RNN-LSTM architectures
- Create robust text preprocessing pipelines for news content
- Address ethical considerations in ML-based content moderation

#### Dataset
- **Primary Dataset**: 44,898 samples from Kaggle Fake News Detection Dataset
- **Testing Dataset**: 5,200 samples from UTK Machine Learning Club competition
- **Features**: Title, Text, Subject, Date
- **Classes**: Real (Reuters.com) vs Fake (flagged by Politifact/Wikipedia)

#### Model Performance
**External Test Dataset Results:**

| Model | Accuracy | Precision (Real) | Precision (Fake) | Recall (Real) | Recall (Fake) | F1 (Real) | F1 (Fake) |
|-------|----------|------------------|------------------|---------------|---------------|-----------|-----------|
| CNN | 0.60 | 0.67 | 0.59 | 0.21 | 0.91 | 0.33 | 0.72 |
| RNN-LSTM | 0.57 | 0.53 | 0.59 | 0.42 | 0.69 | 0.47 | 0.64 |

**Key Findings:**
- CNN achieved higher overall accuracy but struggled with real news detection
- RNN-LSTM showed more balanced performance across classes
- Both models exhibited significant overfitting (99-100% training accuracy vs lower test performance)
- Demonstrates the challenge of generalization in fake news detection

#### Technical Implementation
- **Frameworks**: TensorFlow/Keras, scikit-learn, NLTK
- **Preprocessing**: Text normalization, tokenization, lemmatization, padding
- **CNN Architecture**: 1D convolutions with max pooling (19M parameters)
- **RNN-LSTM Architecture**: Bidirectional LSTM with dropout (18M parameters)

## Repository Structure
```
├── NLP AE1/
│   └── NLP AE1.ipynb                # Sentiment Analysis: NB vs BiLSTM
├── NLP AE2/
│   ├── NLP AE2.pdf                  # Complete research paper and methodology
│   ├── NLP AE2 CNN Final.ipynb      # Final CNN implementation
│   ├── NLP AE2 RNN Final.ipynb      # Final RNN-LSTM implementation
│   └── NLP AE2 CNN with dates.ipynb # Original CNN with temporal features
└── README.md                        # Project documentation
```

## Technical Stack
- **Languages**: Python 3
- **Deep Learning**: TensorFlow/Keras
- **ML Libraries**: scikit-learn, pandas, NumPy
- **NLP Tools**: NLTK, SpaCy
- **Visualization**: matplotlib
- **Development**: Jupyter Notebooks

## Key Methodologies
- **Text Preprocessing**: Comprehensive pipelines for data cleaning and normalization
- **Feature Engineering**: BoW vectorization, TF-IDF, word embeddings
- **Model Architecture**: CNN 1D convolutions, Bidirectional LSTM, attention mechanisms
- **Regularization**: Dropout, L2 regularization, early stopping
- **Evaluation**: Cross-validation, confusion matrices, precision-recall analysis

## Research Contributions
- **Comparative Analysis**: Systematic evaluation of traditional vs deep learning approaches
- **Overfitting Analysis**: Detailed examination of generalization challenges in NLP
- **Ethical Framework**: Discussion of bias, transparency, and responsible AI deployment
- **Preprocessing Innovation**: Robust text processing pipelines for news and review data
- **Performance Benchmarking**: Comprehensive metrics across multiple model architectures

## Future Directions
- **Advanced Architectures**: Transformer models (BERT, RoBERTa) implementation
- **Hybrid Approaches**: Combining CNN and LSTM advantages
- **Feature Enhancement**: Source credibility and metadata incorporation
- **Ethical AI**: Improved bias detection and mitigation strategies
- **Scalability**: Cloud-based distributed training implementation

## Ethical Considerations
- **Algorithmic Bias**: Risk assessment and mitigation strategies
- **Content Moderation**: Transparency in automated decision-making
- **Data Privacy**: Protection protocols for sensitive text data
- **Societal Impact**: Evaluation of AI systems on democratic processes

**Disclaimer**: This research is for academic purposes only. Models are not intended for production content moderation without extensive validation and human oversight. Always consider multiple sources and professional fact-checking for critical information verification.

---

*Master's coursework demonstrating advanced NLP techniques, comparative model analysis, and ethical AI considerations in natural language processing applications.*
