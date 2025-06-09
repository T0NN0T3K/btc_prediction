import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import time
import schedule
import logging
import os
from pathlib import Path

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('btc_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BTCPredictor:
    def __init__(self):
        # Parametri configurabili
        self.INPUT_SEQ_LENGTH = 28    # Quanti giorni usare come input
        self.PREDICTION_HORIZON = 1  # Quanti giorni nel futuro prevedere
        self.ENSEMBLE_SIZE = 3       # Numero di modelli nell'ensemble
        self.CSV_FILE_PATH = 'btc_input_data.csv'
        self.EXECUTION_TIME = "00:30"  # Orario di esecuzione giornaliera (UTC)
        
        # Variabili per il controllo dell'esecuzione
        self.last_prediction_date = None
        self.models = None
        self.scaler = None
        
    def load_and_preprocess_data(self):
        """Carica e preprocessa i dati"""
        try:
            if not os.path.exists(self.CSV_FILE_PATH):
                logger.error(f"File {self.CSV_FILE_PATH} non trovato!")
                return None
                
            # Carica il CSV
            df = pd.read_csv(self.CSV_FILE_PATH)
            
            # Converte timestamp in datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Ordina per data
            df = df.sort_values('timestamp')
            
            # Gestione valori nulli
            df = df.dropna(subset=['BTC_Close'])
            
            # Feature da utilizzare
            features = ['BTC_Close', 'BTC_Volume', 'RSI_14d', 'Fear_Greed_Index', 
                       'Hash_Rate', 'Mining_Difficulty', 'USDX', 'S&P500', 'EFFR', 'Gold']
            
            # Riempi i valori nulli
            for col in features:
                if col == 'BTC_Close' or col == 'BTC_Volume':
                    continue
                elif col == 'Fear_Greed_Index':
                    df[col] = df[col].fillna(50)
                else:
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"Dati caricati: {len(df)} righe dal {df['timestamp'].min()} al {df['timestamp'].max()}")
            return df
            
        except Exception as e:
            logger.error(f"Errore nel caricamento dati: {e}")
            return None

    def create_sequences(self, data, input_seq_length, prediction_horizon=1):
        """Crea sequenze temporali per l'addestramento"""
        X, y = [], []
        for i in range(len(data) - input_seq_length - prediction_horizon + 1):
            sequence = data[i:(i + input_seq_length)]
            target = data[i + input_seq_length + prediction_horizon - 1, 0]  # BTC_Close
            if np.isnan(sequence).any() or np.isnan(target):
                continue
            X.append(sequence)
            y.append(target)
        return np.array(X), np.array(y)

    def create_ensemble_models(self, X, y, feature_count=10):
        """Crea e addestra ensemble di modelli"""
        models = []
        
        for i in range(self.ENSEMBLE_SIZE):
            model = Sequential([
                LSTM(100, return_sequences=True, input_shape=(self.INPUT_SEQ_LENGTH, feature_count)),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=5, restore_best_weights=True
            )
            
            # Addestramento silenzioso
            history = model.fit(
                X, y,
                epochs=50,
                batch_size=32,
                verbose=0,  # Silenzioso per evitare troppo output
                callbacks=[early_stopping]
            )
            
            models.append(model)
            logger.info(f"Modello {i+1}/{self.ENSEMBLE_SIZE} addestrato")
        
        return models

    def save_prediction_to_csv(self, df, predicted_date, predicted_price, prediction_std):
        """Salva la predizione nel CSV"""
        try:
            df_updated = df.copy()
            
            # Assicurati che le colonne esistano
            if 'BTC_Predicted' not in df_updated.columns:
                df_updated['BTC_Predicted'] = np.nan
            if 'BTC_Predicted_Std' not in df_updated.columns:
                df_updated['BTC_Predicted_Std'] = np.nan
            
            # Converti predicted_date in datetime se necessario
            if isinstance(predicted_date, str):
                predicted_date = pd.to_datetime(predicted_date)
            
            # Cerca riga esistente
            existing_row_idx = df_updated[df_updated['timestamp'] == predicted_date].index
            
            if len(existing_row_idx) > 0:
                # Aggiorna riga esistente
                df_updated.loc[existing_row_idx[0], 'BTC_Predicted'] = predicted_price
                df_updated.loc[existing_row_idx[0], 'BTC_Predicted_Std'] = prediction_std
                logger.info(f"Aggiornata predizione per {predicted_date.strftime('%Y-%m-%d')}: ${predicted_price:.2f} (±${prediction_std:.2f})")
            else:
                # Crea nuova riga
                new_row = {col: np.nan for col in df_updated.columns}
                new_row['timestamp'] = predicted_date
                new_row['BTC_Predicted'] = predicted_price
                new_row['BTC_Predicted_Std'] = prediction_std
                
                df_updated = pd.concat([df_updated, pd.DataFrame([new_row])], ignore_index=True)
                df_updated = df_updated.sort_values('timestamp').reset_index(drop=True)
                
                logger.info(f"Aggiunta predizione per {predicted_date.strftime('%Y-%m-%d')}: ${predicted_price:.2f} (±${prediction_std:.2f})")
            
            # Salva il file
            df_updated.to_csv(self.CSV_FILE_PATH, index=False)
            logger.info(f"File salvato: {self.CSV_FILE_PATH}")
            
            return df_updated
            
        except Exception as e:
            logger.error(f"Errore nel salvare predizione: {e}")
            return df

    def should_make_prediction(self, df):
        """Controlla se dovrebbe fare una predizione oggi"""
        try:
            # Data di oggi in UTC
            today_utc = datetime.now(pytz.UTC).date()
            
            # Ultima data nei dati
            last_data_date = df['timestamp'].iloc[-1].date()
            
            # Data della prossima predizione
            next_prediction_date = last_data_date + timedelta(days=self.PREDICTION_HORIZON)
            
            # Controlla se abbiamo già fatto questa predizione
            if 'BTC_Predicted' in df.columns:
                existing_predictions = df[df['BTC_Predicted'].notna()]
                if len(existing_predictions) > 0:
                    last_prediction_for_date = existing_predictions['timestamp'].iloc[-1].date()
                    if last_prediction_for_date >= next_prediction_date:
                        logger.info(f"Predizione già esistente per {next_prediction_date}")
                        return False, None
            
            # Fai predizione se siamo nel giorno giusto o oltre
            if today_utc >= next_prediction_date:
                return True, next_prediction_date
            else:
                logger.info(f"Aspetto fino al {next_prediction_date} per fare la predizione")
                return False, None
                
        except Exception as e:
            logger.error(f"Errore nel controllo predizione: {e}")
            return False, None

    def make_prediction(self):
        """Esegue una predizione completa"""
        try:
            logger.info("🚀 Inizio processo di predizione...")
            
            # Carica dati
            df = self.load_and_preprocess_data()
            if df is None:
                return False
            
            # Controlla se dovrebbe fare predizione
            should_predict, prediction_date = self.should_make_prediction(df)
            if not should_predict:
                return True  # Non è un errore, semplicemente non è il momento
            
            logger.info(f"📊 Preparazione predizione per {prediction_date}")
            
            # Seleziona features
            features = ['BTC_Close', 'BTC_Volume', 'RSI_14d', 'Fear_Greed_Index', 
                       'Hash_Rate', 'Mining_Difficulty', 'USDX', 'S&P500', 'EFFR', 'Gold']
            
            # Normalizzazione
            self.scaler = MinMaxScaler()
            scaled_data = self.scaler.fit_transform(df[features])
            
            # Crea sequenze
            X, y = self.create_sequences(scaled_data, self.INPUT_SEQ_LENGTH, self.PREDICTION_HORIZON)
            
            if len(X) == 0:
                logger.error("Nessuna sequenza valida generata!")
                return False
            
            logger.info(f"Dataset: {len(X)} sequenze di training")
            
            # Addestra modelli
            logger.info("🤖 Addestramento ensemble di modelli...")
            self.models = self.create_ensemble_models(X, y, len(features))
            
            # Prepara predizione
            last_sequence = scaled_data[-self.INPUT_SEQ_LENGTH:]
            last_sequence = last_sequence.reshape(1, self.INPUT_SEQ_LENGTH, len(features))
            
            # Predizioni da ensemble
            predictions = []
            for i, model in enumerate(self.models):
                pred = model.predict(last_sequence, verbose=0)
                dummy_array = np.zeros((1, len(features)))
                dummy_array[0, 0] = pred[0, 0]
                pred_denorm = self.scaler.inverse_transform(dummy_array)[0, 0]
                predictions.append(pred_denorm)
            
            # Calcola statistiche
            mean_prediction = np.mean(predictions)
            std_prediction = np.std(predictions)
            
            # Calcola data predizione normalizzata
            predicted_date = pd.to_datetime(prediction_date).normalize()
            
            # Calcola variazione percentuale
            last_price = df['BTC_Close'].iloc[-1]
            percent_change = ((mean_prediction - last_price) / last_price) * 100
            
            # Log risultati
            logger.info(f"💰 Ultimo prezzo: ${last_price:.2f}")
            logger.info(f"🎯 Predizione per {predicted_date.strftime('%Y-%m-%d')}: ${mean_prediction:.2f} (±${std_prediction:.2f})")
            logger.info(f"📈 Variazione prevista: {percent_change:.2f}%")
            
            # Salva predizione
            df_updated = self.save_prediction_to_csv(df, predicted_date, mean_prediction, std_prediction)
            
            logger.info("✅ Predizione completata con successo!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Errore durante predizione: {e}")
            return False

    def run_continuous(self):
        """Esegue il processo in modo continuo"""
        logger.info(f"🔄 Avvio BTC Predictor in modalità continua")
        logger.info(f"⏰ Esecuzione programmata alle {self.EXECUTION_TIME} UTC ogni giorno")
        logger.info(f"📋 Configurazione: {self.INPUT_SEQ_LENGTH} giorni input, {self.PREDICTION_HORIZON} giorni horizon")
        
        # Programma l'esecuzione giornaliera
        schedule.every().day.at(self.EXECUTION_TIME).do(self.make_prediction)
        
        # Esegui immediatamente la prima volta
        logger.info("🚀 Esecuzione iniziale...")
        self.make_prediction()
        
        # Loop continuo
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Controlla ogni minuto
                
                # Log periodico per mostrare che è attivo
                current_time = datetime.now(pytz.UTC)
                if current_time.minute == 0:  # Una volta all'ora
                    logger.info(f"💓 Sistema attivo - {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                    
            except KeyboardInterrupt:
                logger.info("🛑 Interruzione manuale ricevuta")
                break
            except Exception as e:
                logger.error(f"❌ Errore nel loop principale: {e}")
                time.sleep(300)  # Aspetta 5 minuti prima di riprovare

def main():
    """Funzione principale"""
    try:
        predictor = BTCPredictor()
        predictor.run_continuous()
    except Exception as e:
        logger.error(f"❌ Errore fatale: {e}")

if __name__ == "__main__":
    main()