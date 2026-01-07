fake_news_detector/
├── models/
│   ├── __init__.py
│   ├── base_model.py           # Base abstract class
│   ├── bert_detector.py        # BERT-based detector
│   ├── roberta_detector.py     # RoBERTa-based detector
│   ├── tfidf_detector.py       # TF-IDF + classifier
│   └── ensemble_detector.py    # Ensemble combining all models
├── services/
│   ├── __init__.py
│   ├── inference.py           # Main inference service
│   └── explainer.py           # Explanation generation (SHAP, LIME)
├── utils/
│   ├── __init__.py
│   └── credibility_scorer.py  # Source credibility scoring
└── config.py                  # Configuration
