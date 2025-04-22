from flask import Flask, jsonify, request
from flask_cors import CORS 
from db import get_db_connection
import os
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter


# Load the saved model
ecg_model = keras.models.load_model("ecg_afib_model.h5")
emg_model = keras.models.load_model("emg_classifier.h5")
eog_model = keras.models.load_model("eog_sleep_stage_model.h5")
eeg_model = keras.models.load_model("eeg_sleep_stage_model.h5")


def load_emg_data(file_path, label, segment_length=200):
    data = np.loadtxt(file_path)[:, 1]  # Extract signal values only
    num_segments = len(data) // segment_length  # Number of segments
    data = data[:num_segments * segment_length]  # Trim excess
    segmented_data = data.reshape(num_segments, segment_length)  # Reshape into (samples, time_steps)
    return segmented_data, np.full(num_segments, label)  # Labels for each segment

# Load and segment data
X_healthy, y_healthy = load_emg_data('emg_dataset/emg_healthy.txt', 'healthy')
X_myopathy, y_myopathy = load_emg_data('emg_dataset/emg_myopathy.txt', 'myopathy')
X_neuropathy, y_neuropathy = load_emg_data('emg_dataset/emg_neuropathy.txt', 'neuropathy')

# Combine datasets
X = np.vstack([X_healthy, X_myopathy, X_neuropathy])
y = np.concatenate([y_healthy, y_myopathy, y_neuropathy])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalize input data
scaler = StandardScaler()
X = scaler.fit_transform(X)

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def home():
    return "AI Server"

# # Route to get all users
@app.route('/get_users', methods=['GET'])
def get_users():
    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch all users from the users table
    cur.execute("SELECT * FROM users;")
    users = cur.fetchall()

    cur.close()
    conn.close()

    # Format users into a list of dictionaries
    return jsonify([{"id": user[0], "username": user[1], "email": user[2]} for user in users])

# Atrial Fibrillation detection function
segment_size = 3000
def predict_ecg_segment(segment):
    segment = np.array(segment).reshape(1, segment_size, 1)  # Reshape for CNN
    prediction = ecg_model.predict(segment)
    return "Atrial Fibrillation" if prediction > 0.5 else "Normal"


def predict_emg_segment(data, segment_length=200):    
    num_segments = len(data) // segment_length  # Ensure correct segmentation
    data = data[:num_segments * segment_length]  # Trim excess
    segmented_data = data.reshape(num_segments, segment_length)  # Reshape into segments
    
    # Normalize
    segmented_data = scaler.transform(segmented_data)
    segmented_data = segmented_data.reshape(segmented_data.shape[0], segmented_data.shape[1], 1)
    
    predictions = emg_model.predict(segmented_data)
    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    
    unique, counts = np.unique(predicted_labels, return_counts=True)
    total_predictions = len(predicted_labels)
    percentage_results = {label: (count / total_predictions) * 100 for label, count in zip(unique, counts)}
    
    categories = ['healthy', 'myopathy', 'neuropathy']
    #final_results = {category: str(round(percentage_results.get(category, 0.0),2))+"%"  for category in categories}
    final_results = {"healthy": str(round(percentage_results.get("healthy", 0.0),2)), "myopathy": str(round(percentage_results.get("myopathy", 0.0),2)),
                     "neuropathy": str(round(percentage_results.get("neuropathy", 0.0),2)),
                     }
    return final_results


def predict_ecg_prob(data):
    hit = 0
    total = int(len(data)/segment_size)
    for i in range(0,len(data),segment_size):
        if predict_ecg_segment(data[i:i+segment_size]) == "Normal":
            hit += 1
    
    prob_normal = hit/total*100
    prob_afib = (total-hit)/total*100
    return {"Normal":str(prob_normal)+"%","AFib":str(prob_afib)+"%"}


def map_array(values, in_min, in_max, out_min, out_max):
    return np.interp(values, [in_min, in_max], [out_min, out_max])

def pre_process_data(array_data):
    signal_data = [data[0] for data in array_data]
    signal_data.reverse()
    signal_data = map_array(signal_data, -4096, 4096, -10, 10)
    return signal_data



def predict_eog_segment(eog_data):
    new_eog_data = np.array(eog_data)  # Shape: (N,)
    sampling_rate = 100  # Example: 100 Hz (Modify if needed)
    # Define epoch duration
    epoch_duration = 30  # Each epoch = 30 seconds
    # Calculate the number of samples per epoch
    samples_per_epoch = sampling_rate * epoch_duration  # E.g., 100 * 30 = 3000
    # Ensure data is trimmed to a multiple of samples_per_epoch
    num_epochs = new_eog_data.shape[0] // samples_per_epoch
    new_eog_data = new_eog_data[: num_epochs * samples_per_epoch]  # Trim extra samples
    # Reshape into CNN-compatible format: (num_epochs, samples_per_epoch, 1)
    X_new = new_eog_data.reshape(num_epochs, samples_per_epoch, 1)
    # Predict sleep stages
    y_pred_new = np.argmax(eog_model.predict(X_new), axis=1)

    # Map numeric predictions to sleep stage names
    stage_labels = {0: "Awake", 1: "NREM", 2: "REM"}  # Modify based on your labels
    stage_counts = Counter(y_pred_new)

    # Convert to readable format
    stage_durations = {stage_labels[k]: v for k, v in stage_counts.items()}
        
    epoch_duration = 30  # seconds
    stage_durations_minutes = {stage: (epochs * epoch_duration) / 60 for stage, epochs in stage_durations.items()}

    result = {stage: f"{duration:.2f} minutes" for stage, duration in stage_durations_minutes.items()}
    return result

def predict_eeg_segment(eeg_data):
    new_eeg_data = np.array(eeg_data)  # Shape: (N,)
    sampling_rate = 100  # Example: 100 Hz (Modify if needed)
    # Define epoch duration
    epoch_duration = 30  # Each epoch = 30 seconds
    # Calculate the number of samples per epoch
    samples_per_epoch = sampling_rate * epoch_duration  # E.g., 100 * 30 = 3000
    # Ensure data is trimmed to a multiple of samples_per_epoch
    num_epochs = new_eeg_data.shape[0] // samples_per_epoch
    new_eeg_data = new_eeg_data[: num_epochs * samples_per_epoch]  # Trim extra samples
    # Reshape into CNN-compatible format: (num_epochs, samples_per_epoch, 1)
    X_new = new_eeg_data.reshape(num_epochs, samples_per_epoch, 1)
    # Predict sleep stages
    y_pred_new = np.argmax(eeg_model.predict(X_new), axis=1)

    # Map numeric predictions to sleep stage names
    stage_labels = {0: "Awake", 1: "NREM", 2: "REM"}  # Modify based on your labels
    stage_counts = Counter(y_pred_new)

    # Convert to readable format
    stage_durations = {stage_labels[k]: v for k, v in stage_counts.items()}
        
    epoch_duration = 30  # seconds
    stage_durations_minutes = {stage: (epochs * epoch_duration) / 60 for stage, epochs in stage_durations.items()}

    result = {stage: f"{duration:.2f} minutes" for stage, duration in stage_durations_minutes.items()}
    return result


@app.route('/ecg_analysis', methods=['GET'])
def get_ecg_result():
    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch all users from the users table
    cur.execute("SELECT data FROM ecg ORDER BY id DESC LIMIT 12000;") #250 sampling rate. 15000 for 60 seconds
    ecg_data = cur.fetchall()

    ecg_data = pre_process_data(ecg_data)
    
    #ecg_result = predict_ecg_segment(ecg_data)
    ecg_result = predict_ecg_prob(ecg_data)
    cur.close()
    conn.close()
    
    # Format users into a list of dictionaries
    return jsonify(ecg_result)

@app.route('/emg_analysis', methods=['GET'])
def get_emg_result():
    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch all users from the users table
    cur.execute("SELECT data FROM emg ORDER BY id DESC LIMIT 12000;") #250 sampling rate. 15000 for 60 seconds
    emg_data = cur.fetchall()

    emg_data = pre_process_data(emg_data)
    
    emg_result = predict_emg_segment(emg_data)
    cur.close()
    conn.close()
    
    # Format users into a list of dictionaries
    return jsonify(emg_result)

@app.route('/eog_analysis', methods=['GET'])
def get_eog_result():
    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch all users from the users table
    cur.execute("SELECT data FROM eog ORDER BY id DESC LIMIT 12000;") #250 sampling rate. 15000 for 60 seconds
    eog_data = cur.fetchall()

    eog_data = pre_process_data(eog_data)
    
    eog_result = predict_eog_segment(eog_data)
    cur.close()
    conn.close()
    
    # Format users into a list of dictionaries
    return jsonify(eog_result)

@app.route('/eeg_analysis', methods=['GET'])
def get_eeg_result():
    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch all users from the users table
    cur.execute("SELECT data FROM eeg ORDER BY id DESC LIMIT 12000;") #250 sampling rate. 15000 for 60 seconds
    eeg_data = cur.fetchall()

    eeg_data = pre_process_data(eeg_data)
    eeg_result = {}
    try:
        eeg_result = predict_eeg_segment(eeg_data)
    except Exception:
        eeg_result = {"Awake":"error", "NREM":"error","REM":"error"}
    cur.close()
    conn.close()
    
    # Format users into a list of dictionaries
    return jsonify(eeg_result)


@app.route('/temp_analysis', methods=['GET'])
def get_temp_result():
    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch all users from the users table
    cur.execute("SELECT AVG(data) AS avg_temp_data FROM (SELECT data FROM temperature ORDER BY id DESC LIMIT 10) subquery;")
    temp_data = cur.fetchall()
    temp_data = float(temp_data[0][0])
    
    temp_analysis = ""
    if(temp_data <= 37.7):
        temp_analysis = "normal"
    elif (temp_data > 37.7 and temp_data <= 39.3 ):
        temp_analysis = "moderate fever"
    elif (temp_data > 39.3):
        temp_analysis = "high fever"
    
    temp_result = {"avg_temp": temp_data, "temp_analysis": temp_analysis }
    cur.close()
    conn.close()
    
    # Format users into a list of dictionaries
    return jsonify(temp_result)


@app.route("/api/data/<type>", methods=["GET"])
def get_data(type):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query_map = {
            "eeg": "SELECT id, device_topic, device_id, data, timestamp FROM eeg ORDER BY id DESC LIMIT 3000",
            "ecg": "SELECT id, device_topic, device_id, data, timestamp FROM ecg ORDER BY id DESC LIMIT 3000",
            "emg": "SELECT id, device_topic, device_id, data, timestamp FROM emg ORDER BY id DESC LIMIT 3000",
            "eog": "SELECT id, device_topic, device_id, data, timestamp FROM eog ORDER BY id DESC LIMIT 3000",
            "temperature": "SELECT id, device_topic, device_id, data, timestamp FROM temperature ORDER BY id DESC LIMIT 300",
            "spo2": "SELECT id, device_topic, device_id, data, timestamp FROM spo2 ORDER BY id DESC LIMIT 3000"
        }
        
        
        query = query_map.get(type)
        if query:
            cursor.execute(query)
            data = cursor.fetchall()  # Already a list of dictionaries
            data = [{"data": i[3], "timestamp": i[4]} for i in data]
            return jsonify({"result": True, "data": data})
        else:
            return jsonify({"result": False})
    except Exception as e:
        return jsonify({"result": False, "error": str(e)})
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
