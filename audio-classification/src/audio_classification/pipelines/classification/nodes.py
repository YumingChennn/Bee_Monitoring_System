# nodes.py
import os
import io
import pymongo
import gridfs
import pickle
import numpy as np
import librosa
import soundfile as sf
from scipy import stats

# Step 1: Fetch audio files from MongoDB
def fetch_audio_files_from_mongodb(limit, mongodb_uri, db_name, collection_name):
    client = pymongo.MongoClient(mongodb_uri)
    db = client[db_name]
    fs = gridfs.GridFS(db)
    documents = db[collection_name].find().limit(limit)
    
    audio_files = []
    audio_ids = []  # To keep track of the audio file IDs

    for doc in documents:
        audio_id = doc['audio_file_id']
        audio_data = fs.get(audio_id).read()  # Retrieve audio file
        audio_bytes = io.BytesIO(audio_data)
        # Use soundfile or librosa to read audio file
        y, sr = sf.read(audio_bytes)
        audio_files.append((y, sr))
        audio_ids.append(audio_id)

    print(f"Total audio files fetched: {len(audio_files)}")

    if not audio_files:
        raise ValueError("fetch_audio_files_from_mongodb error No audio files provided to split into chunks.")
    return audio_files, audio_ids

# Step 2: Split audio into chunks
def split_audio_chunks(audio_files, chunk_size, hop_size):
    print("Splitting audio into chunks...")
    chunks = []

    for y, sr in audio_files:  # Here audio_files contains audio data and sample rate
        chunk_samples = int(chunk_size * sr)
        hop_samples = int(hop_size * sr)
        for start in range(0, len(y) - chunk_samples + 1, hop_samples):
            end = start + chunk_samples
            chunks.append(y[start:end])
    
    print(f"Total chunks created: {len(chunks)}")
    return chunks, sr  # Return sample rate

# Step 3: Load models
def load_models(knn_model_path):
    with open(knn_model_path, 'rb') as file:
        knn_model = pickle.load(file)
    return knn_model

# Step 4: Extract MFCC features
def extract_mfcc_features(data, sample_rate, n_mfcc=20, chunk_size=22050):
    mfcc_features = []
    num_chunks = len(data) // chunk_size
    for i in range(num_chunks):
        chunk = data[i * chunk_size:(i + 1) * chunk_size]
        mfcc = librosa.feature.mfcc(y=chunk, sr=sample_rate, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_features.append(mfcc_mean)
    return np.array(mfcc_features)

# Step 5: Predict audio files using models
def predict_audio_files(audio_chunks, sample_rate, knn_model):
    all_knn_preds = []

    for chunk in audio_chunks:
        # Extract MFCC features
        mfcc_features = extract_mfcc_features(chunk, sample_rate)
        
        # KNN prediction
        knn_pred = knn_model.predict(mfcc_features.reshape(mfcc_features.shape[0], -1))
        all_knn_preds.append(knn_pred)

    # Calculate the final prediction using mode
    final_knn_pred = stats.mode(all_knn_preds, axis=0)[0][0]
    
    print(f'Final KNN Prediction: {final_knn_pred}')
    return final_knn_pred

# Function to predict all audio files
def predict_all_audio_files(audio_files, audio_chunks, sample_rate, knn_model):
    print("Predicting all audio files...")
    predictions = []
    
    for i in range(len(audio_files)):
        prediction = predict_audio_files([audio_chunks[i]], sample_rate, knn_model)
        predictions.append(prediction)
    
    return predictions

def save_predictions_to_mongodb( mongodb_uri, db_name, collection_name, audio_ids, predictions):
    # Connect to MongoDB
    client = pymongo.MongoClient(mongodb_uri)
    db = client[db_name]
    audio_collection = db[collection_name]  # Collection to update predictions
    
    for audio_id, prediction in zip(audio_ids, predictions):
        # Convert numpy int64 to Python int
        prediction_value = int(prediction)  # Ensure it's a native Python int
        # Update the document with the new prediction
        audio_collection.update_one(
            {'audio_file_id': audio_id},  # Filter to find the document
            {'$set': {'prediction': prediction_value}}  # Add or update the prediction field
        )
    print(f"Updated predictions for {len(predictions)} audio files in MongoDB.")