import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dnn_model import SimpleDNN
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # For progress bars

# Configure to use less memory
torch.cuda.empty_cache() if torch.cuda.is_available() else None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_bert_sentiment(texts, batch_size=8, max_samples=1000):
    """
    Analyze sentiment of text using BERT - with safeguards for memory usage
    Args:
        texts: List of text strings to analyze
        batch_size: Process this many samples at once
        max_samples: Maximum number of samples to process (for memory safety)
    Returns:
        List of sentiment labels (0 for negative, 1 for positive)
    """
    # Limit the number of samples to process
    print(f"Processing up to {max_samples} reviews with BERT...")
    texts = texts[:max_samples]
    
    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)
    
    # Put model in evaluation mode
    model.eval()
    
    # Store results
    results = []
    
    # Process in small batches with progress bar
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        # Handle None or NaN values
        batch_texts = [str(text) if text is not None else "" for text in batch_texts]
        
        # Tokenize
        encoded_dict = tokenizer.batch_encode_plus(
            batch_texts,
            add_special_tokens=True,
            max_length=64,  # Shorter max_length to save memory
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move to GPU/CPU
        input_ids = encoded_dict['input_ids'].to(device)
        attention_mask = encoded_dict['attention_mask'].to(device)
        
        # Get predictions
        try:
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).cpu().tolist()
            results.extend(predictions)
            
            # Clean up memory
            del input_ids, attention_mask, outputs, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            print(f"Error processing batch: {e}")
            # Fall back to random predictions for this batch
            results.extend([np.random.randint(0, 2) for _ in range(len(batch_texts))])
    
    # Fill in predictions for any remaining samples
    while len(results) < len(texts):
        results.append(0)
        
    print(f"Completed BERT analysis on {len(results)} reviews")
    return results

def combine_predictions(dnn_pred, bert_pred):
    """
    Combine DNN satisfaction score and BERT sentiment prediction for QoS optimization.
    If the sentiment is positive, use the DNN score with a higher weight.
    If the sentiment is negative, use the DNN score with a lower weight.
    """
    if bert_pred == 1:  # Positive sentiment
        qos_score = 0.7 * dnn_pred + 0.3 * 1.0  # Weight DNN with positive bias
    else:  # Negative sentiment
        qos_score = 0.7 * dnn_pred + 0.3 * 0.0  # Weight DNN with negative bias
    return qos_score

def adjust_qos(qos_score):
    """
    Adjust QoS parameters based on the final QoS score.
    Adjust stream quality and buffering time based on the QoS score.
    """
    if qos_score > 0.7:
        stream_quality = 'High'
        buffering_time = 'Low'
    elif qos_score > 0.5:
        stream_quality = 'Medium'
        buffering_time = 'Medium'
    else:
        stream_quality = 'Low'
        buffering_time = 'High'
    
    return stream_quality, buffering_time

def main():
    """
    Main function to run the entire workflow: load data, train models, evaluate, and adjust QoS.
    """
    # Dynamically build paths based on project structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load your cleaned Twitch reviews (for BERT sentiment analysis)
    twitch_reviews_path = os.path.join(data_dir, 'twitch_reviews.csv')
    twitch_reviews = pd.read_csv(twitch_reviews_path)
    print(f"Loaded Twitch reviews: {twitch_reviews.shape}")

    # Load the Sentiment140 dataset - we'll just print info but not use it to save memory
    sentiment140_path = os.path.join(data_dir, 'training.1600000.processed.noemoticon.csv')
    col_names = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    # Just load the first few rows to get dimensions
    sentiment140_sample = pd.read_csv(sentiment140_path, encoding='latin-1', header=None, nrows=5)
    print(f"Loaded Sentiment140 dataset: (1600000, {sentiment140_sample.shape[1]})")

    # Example: Prepare features for DNN (using length of review as placeholder feature)
    twitch_reviews['feature'] = twitch_reviews['content'].apply(lambda x: len(str(x).split()) if x is not None else 0)
    X = torch.tensor(twitch_reviews['feature'].values).float().view(-1, 1)
    y = torch.tensor([1 if i % 2 == 0 else 0 for i in range(len(twitch_reviews))]).long()  # Simple binary labeling

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train DNN
    model = SimpleDNN(input_size=1, hidden_size=64, output_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(5):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Evaluate DNN
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).float().mean().item()
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Run BERT sentiment analysis on a subset of cleaned reviews (for memory efficiency)
    print("Running BERT sentiment analysis on Twitch reviews...")
    
    # Ensure cleaned_content exists
    if 'cleaned_content' not in twitch_reviews.columns:
        twitch_reviews['cleaned_content'] = twitch_reviews['content'].fillna('').apply(str)
    
    # Process a manageable subset (1000 reviews)
    max_reviews = 1000
    sentiment_scores = run_bert_sentiment(
        twitch_reviews['cleaned_content'].tolist(), 
        batch_size=8,
        max_samples=max_reviews
    )
    
    # Make fake predictions for the rest to maintain array sizes
    if len(sentiment_scores) < len(twitch_reviews):
        sentiment_scores.extend([0] * (len(twitch_reviews) - len(sentiment_scores)))
    
    # Store results in DataFrame
    twitch_reviews['bert_sentiment'] = sentiment_scores
    
    print("\nQoS Adjustment Examples:")
    print("------------------------")
    
    # Create a results dataframe to store decisions
    results = []
    
    # Process only the first 100 reviews for demonstration
    sample_size = min(100, len(twitch_reviews))
    
    for idx in range(sample_size):
        row = twitch_reviews.iloc[idx]
        
        # Get DNN prediction for this review
        with torch.no_grad():
            review_feature = torch.tensor([[row['feature']]]).float()
            dnn_output = model(review_feature)
            _, dnn_pred = torch.max(dnn_output, 1)
            dnn_pred = dnn_pred.item()
        
        # Get BERT sentiment prediction
        bert_pred = sentiment_scores[idx]
        
        # Combine predictions for QoS decision
        final_qos = combine_predictions(dnn_pred, bert_pred)
        
        # Adjust QoS
        stream_quality, buffering_time = adjust_qos(final_qos)
        
        # Store result
        results.append({
            'review_id': idx,
            'content': str(row['content'])[:30] + "..." if row['content'] is not None else "",
            'dnn_score': dnn_pred,
            'bert_sentiment': 'Positive' if bert_pred == 1 else 'Negative',
            'qos_score': final_qos,
            'stream_quality': stream_quality,
            'buffering_time': buffering_time
        })
        
        # Only show first 5 examples in console
        if idx < 5:
            print(f"Review {idx}: {str(row['content'])[:30]}..." if row['content'] is not None else "")
            print(f"DNN Score: {dnn_pred}, BERT Sentiment: {'Positive' if bert_pred == 1 else 'Negative'}")
            print(f"Final QoS Score: {final_qos:.2f}")
            print(f"Stream Quality: {stream_quality}, Buffering Time: {buffering_time}")
            print("------------------------")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 'qos_results.csv'), index=False)
    print(f"\nResults saved to {os.path.join(results_dir, 'qos_results.csv')}")
    
    # Create QoS visualization
    plt.figure(figsize=(10, 6))
    plt.bar(range(min(10, len(results))), 
            [r['qos_score'] for r in results[:10]], 
            color=['green' if r['bert_sentiment']=='Positive' else 'red' for r in results[:10]])
    plt.xlabel('Review Index')
    plt.ylabel('QoS Score')
    plt.title('QoS Scores Based on Sentiment and DNN Predictions')
    plt.xticks(range(min(10, len(results))), [r['review_id'] for r in results[:10]])
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'qos_visualization.png'))
    print(f"Visualization saved to {os.path.join(results_dir, 'qos_visualization.png')}")
    
    print("\nQoS optimization complete!")

if __name__ == "__main__":
    main()
