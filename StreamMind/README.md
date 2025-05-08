
# Real-Time QoS Management for Game Streaming with DNN & BERT

## Overview
This project addresses the challenge of maintaining consistent Quality of Service (QoS) in live video game streaming platforms such as Twitch and YouTube Gaming. It leverages:

- **Deep Neural Networks (DNN)** for satisfaction classification
- **BERT-based sentiment analysis** for dynamic user sentiment and engagement interpretation

The system uses benchmark datasets to simulate real-time QoS management and optimization.

## Project Structure
```
qos-project/
│
├── data/                         # Benchmark datasets
│   ├── processed_twitch_reviews.csv
│   ├── training.1600000.processed.noemoticon.csv
│   └── twitch_reviews.csv
│
├── models/                       # Model definitions and checkpoints
│   └── (add your model files here)
│
├── utils/                        # Utility scripts (data, metrics, visualization)
│   └── (add your utility files here)
│
├── src/                          # Main source code
│   ├── bert_sentiment_analysis.py
│   ├── data_preprocessing.py
│   ├── dnn_model.py
│   └── main.py
│
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## How It Works

### Data Preprocessing
Cleans and tokenizes review and QoS data from benchmark datasets.

### Satisfaction Classification (DNN)
Trains a DNN to predict user satisfaction or QoS class from processed features.

### Sentiment Analysis (BERT)
Uses BERT to analyze user reviews/comments and extract sentiment scores.

### QoS Decision Logic
Combines DNN and BERT outputs to simulate real-time QoS management actions (e.g., adjusting bitrate, buffer size).

### Visualization & Evaluation
Plots metrics and relationships between sentiment, satisfaction, and QoS decisions.

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Place your benchmark datasets in the `data/` folder.

### 3. Preprocess Data
```bash
python src/data_preprocessing.py
```

### 4. Train & Evaluate Models
```bash
python src/main.py
```

### 5. Explore Results
Check the terminal output and any generated plots in the `results/` folder.

## Customization
- Add or modify models in the `models/` folder.
- Add or modify helper functions in `utils/`.
- Update `src/main.py` to change the QoS decision logic or experiment with new features.

## Troubleshooting
- If you encounter memory errors with BERT, reduce the batch size or number of reviews processed at once in `main.py`.
- For import errors, ensure your working directory matches the project root when running scripts.

## Credits
- BERT sentiment analysis via **HuggingFace Transformers**
- DNN modeling with **PyTorch**
- Benchmark datasets: **Sentiment140**, **Twitch reviews**

## License
This project is for academic and research purposes.
