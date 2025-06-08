import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Imposta un seed per garantire risultati riproducibili

# 1. Caricamento e preprocessing dei dati
def load_and_preprocess_data(file_path):
    # Carica il CSV
    df = pd.read_csv(file_path)
    
    # Converte timestamp in datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ordina per data
    df = df.sort_values('timestamp')
    
    # Gestione valori nulli
    # Rimuovi righe con BTC_Close nullo (target variabile)
    df = df.dropna(subset=['BTC_Close'])
    
    # Per le feature numeriche, riempi i valori nulli
    features = ['BTC_Close', 'BTC_Volume', 'RSI_14d', 'Fear_Greed_Index', 
                'Hash_Rate', 'Mining_Difficulty', 'USDX', 'S&P500', 'EFFR', 'Gold']
    
    # Riempi i valori nulli con approcci diversi per colonna
    for col in features:
        if col == 'BTC_Close' or col == 'BTC_Volume':
            # Queste colonne devono essere sempre presenti, nessun riempimento aggiuntivo
            continue
        elif col == 'Fear_Greed_Index':
            # Per Fear_Greed_Index (disponibile dal 2018), usa 50 (valore neutrale) per i dati precedenti
            df[col] = df[col].fillna(50)
        else:
            # Per le altre feature (es. dati finanziari weekend), usa forward fill e poi backward fill
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    # Verifica che non ci siano più valori nulli
    if df[features].isnull().sum().sum() > 0:
        print("Attenzione: alcuni valori nulli persistono!")
        print(df[features].isnull().sum())
    
    return df

# 2. Creazione delle sequenze temporali con horizon variabile
def create_sequences(data, input_seq_length, prediction_horizon=1):
    X, y = [], []
    for i in range(len(data) - input_seq_length - prediction_horizon + 1):
        # Sequenza di input
        sequence = data[i:(i + input_seq_length)]
        # Target: prezzo dopo il periodo di prediction_horizon
        target = data[i + input_seq_length + prediction_horizon - 1, 0]  # BTC_Close
        # Salta se ci sono nan nella sequenza o nel target
        if np.isnan(sequence).any() or np.isnan(target):
            continue
        X.append(sequence)
        y.append(target)
    return np.array(X), np.array(y)

# 3. Creazione e addestramento di modelli multipli per ensemble
def create_ensemble_models(X, y, num_models=5, input_seq_length=7, feature_count=10):
    models = []
    
    for i in range(num_models):
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(input_seq_length, feature_count)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Callback per early stopping per evitare overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=5, restore_best_weights=True
        )
        
        # Addestramento
        history = model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            verbose=1,
            callbacks=[early_stopping]
        )
        
        models.append(model)
        print(f"Modello {i+1}/{num_models} addestrato")
    
    return models

# 4. Main execution
def main():
    # Parametri configurabili
    INPUT_SEQ_LENGTH = 28    # Quanti giorni usare come input (finestra di osservazione)
    PREDICTION_HORIZON = 1  # Quanti giorni nel futuro prevedere (default: 1 giorno)
    ENSEMBLE_SIZE = 3       # Numero di modelli nell'ensemble
    
    print(f"Configurazione: Usando {INPUT_SEQ_LENGTH} giorni di dati per prevedere {PREDICTION_HORIZON} giorni nel futuro")
    
    # Carica e preprocessa i dati
    df = load_and_preprocess_data('btc_input_data.csv')
    
    # Seleziona le feature
    features = ['BTC_Close', 'BTC_Volume', 'RSI_14d', 'Fear_Greed_Index', 
                'Hash_Rate', 'Mining_Difficulty', 'USDX', 'S&P500', 'EFFR', 'Gold']
    
    # Normalizzazione
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    # Creazione sequenze per training con horizon variabile
    X, y = create_sequences(scaled_data, INPUT_SEQ_LENGTH, PREDICTION_HORIZON)
    
    if len(X) == 0:
        raise ValueError("Nessuna sequenza valida generata. Controlla i dati!")
    
    print(f"Dataset creato: {len(X)} sequenze di training")
    
    # Crea e addestra ensemble di modelli
    models = create_ensemble_models(X, y, num_models=ENSEMBLE_SIZE, 
                                   input_seq_length=INPUT_SEQ_LENGTH, feature_count=len(features))
    
    # Preparazione per la predizione
    last_sequence = scaled_data[-INPUT_SEQ_LENGTH:]
    last_sequence = last_sequence.reshape(1, INPUT_SEQ_LENGTH, len(features))
    
    # Predizione da ciascun modello nell'ensemble
    predictions = []
    for i, model in enumerate(models):
        pred = model.predict(last_sequence, verbose=0)
        dummy_array = np.zeros((1, len(features)))
        dummy_array[0, 0] = pred[0, 0]
        pred_denorm = scaler.inverse_transform(dummy_array)[0, 0]
        predictions.append(pred_denorm)
        print(f"Modello {i+1}: ${pred_denorm:.2f}")
    
    # Calcola la media e la deviazione standard delle predizioni
    mean_prediction = np.mean(predictions)
    std_prediction = np.std(predictions)
    min_prediction = np.min(predictions)
    max_prediction = np.max(predictions)
    
    # Data della predizione
    last_date = df['timestamp'].iloc[-1]
    predicted_date = last_date + pd.Timedelta(days=PREDICTION_HORIZON)
    
    # Calcola la variazione percentuale
    last_price = df['BTC_Close'].iloc[-1]
    percent_change = ((mean_prediction - last_price) / last_price) * 100
    
    print(f"\nUltimo prezzo conosciuto ({last_date.strftime('%Y-%m-%d')}): ${last_price:.2f}")
    print(f"Prezzo predetto per {predicted_date.strftime('%Y-%m-%d')} (fra {PREDICTION_HORIZON} giorni): ${mean_prediction:.2f}")
    print(f"Intervallo di previsione: ${min_prediction:.2f} - ${max_prediction:.2f}")
    print(f"Deviazione standard: ${std_prediction:.2f}")
    print(f"Variazione percentuale prevista: {percent_change:.2f}%")
    
if __name__ == "__main__":
    main()