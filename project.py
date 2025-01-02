import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
import joblib


# Caricare i dati dai file Excel
file_path_2022 = 'ChallengePOLITO_MastriniMagazzino_2022_1.xls'
df_2022 = pd.read_excel(file_path_2022, engine='xlrd')

file_path_2023 = 'ChallengePOLITO_MastriniMagazzino_2023_1.xls'
df_2023 = pd.read_excel(file_path_2023, engine='xlrd')

file_path_2024 = 'ChallengePOLITO_MastriniMagazzino_2024_1.xls'
df_2024 = pd.read_excel(file_path_2024, engine='xlrd')

# Concatenare i DataFrame
df = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)

# Tenere solo le label che mi interessano
df = df[['product_value', 'product_name', 'uom', 'warehouse', 'movementdate', 'initialstock', 'qty', 'qtyload', 'qtyunload', 'finalstock']]

# Calcolare il numero totale di articoli venduti per ciascun prodotto
df['total_sold'] = df['qty'].apply(lambda x: -x if x < 0 else 0)
top_10_products = df.groupby('product_name')['total_sold'].sum().nlargest(10).index

# Filtrare il dataframe per includere solo i 10 prodotti più venduti
df_top_10 = df[df['product_name'].isin(top_10_products)]

# Feature temporali: mese, giorno della settimana, stagione, settimana dell'anno
df_top_10['month'] = pd.to_datetime(df_top_10['movementdate']).dt.month
df_top_10['day_of_week'] = pd.to_datetime(df_top_10['movementdate']).dt.dayofweek
df_top_10['season'] = df_top_10['month'].apply(lambda x: (x % 12 + 3) // 3)  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
df_top_10['week_of_year'] = pd.to_datetime(df_top_10['movementdate']).dt.isocalendar().week

# Label encoding per product_name
label_encoder = LabelEncoder()
df_top_10['product_encoded'] = label_encoder.fit_transform(df_top_10['product_name'])

# Creare un set di feature
X = df_top_10[['product_encoded', 'month', 'day_of_week', 'season', 'week_of_year', 'initialstock', 'finalstock']]

# Log Transformation per ridurre l'impatto dei valori estremi sul target
y = np.log1p(df_top_10['qty'])

# Rimuovere outlier
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
filter = (y >= Q1 - 1.5 * IQR) & (y <= Q3 + 1.5 * IQR)
X, y = X[filter], y[filter]

# Dividere i dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizzazione delle feature
scaler_X = QuantileTransformer(output_distribution='normal')
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Normalizzare il target
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
# Dopo aver fittato scaler_y
joblib.dump(scaler_y, 'scaler_y.pkl')
# Salva lo scaler_X fittato
joblib.dump(scaler_X, 'scaler_X.pkl')

# Model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='linear'))
# Compilare il modello
#loss_function = Huber(delta=1.0)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
# Salva il modello addestrato
model.save('model.h5')
print("Modello salvato come 'model.h5'")

# Callbacks per ridurre dinamicamente il learning rate
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

# Addestrare il modello
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[lr_scheduler])

# Prevedere sui dati di test
#y_pred = model.predict(X_test).flatten()
y_pred = model.predict(X_test[:12]).flatten()
y_test = y_test[:12]

# Denormalizzare le previsioni
y_pred_denorm = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_denorm = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Invertire la trasformazione logaritmica
y_test_denorm_exp = np.expm1(y_test_denorm)
y_pred_denorm_exp = np.expm1(y_pred_denorm)

print(y_pred_denorm_exp)
print(y_test_denorm_exp)
# Calcolare le metriche
mae = mean_absolute_error(y_test_denorm_exp, y_pred_denorm_exp)
mse = mean_squared_error(y_test_denorm_exp, y_pred_denorm_exp)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test_denorm_exp - y_pred_denorm_exp) / y_test_denorm_exp)) * 100

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAPE: {mape}%')

# Visualizzazione della perdita durante l'addestramento
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#plt.show()

# Plot delle previsioni rispetto ai valori reali
plt.figure(figsize=(12, 6))
plt.scatter(y_test_denorm_exp, y_pred_denorm_exp)
plt.plot([min(y_test_denorm_exp), max(y_test_denorm_exp)], [min(y_test_denorm_exp), max(y_test_denorm_exp)], color='red', linestyle='--')
plt.title('Predictions vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
#plt.show()

# Grafico a linee delle vendite dei prodotti nei vari mesi
plt.figure(figsize=(12, 6))
for product in top_10_products:
    monthly_sales = df_top_10[df_top_10['product_name'] == product].groupby('month')['qty'].sum()
    plt.plot(monthly_sales.index, monthly_sales.values, marker='o', label=product)

plt.title('Vendite Mensili dei Prodotti')
plt.xlabel('Mese')
plt.ylabel('Quantità Venduta')
plt.legend()
#plt.show()



# Funzione per prevedere la quantità necessaria
def predict_quantity(product_name, month):
    # Encoding del nome del prodotto
    product_encoded = label_encoder.transform([product_name])[0]
    
    # Calcoli per le feature mancanti
    day_of_week = 0  # Assumiamo lunedì (o un valore neutro)
    season = (month % 12 + 3) // 3
    week_of_year = (month - 1) * 4 + 2  # Approssimazione per la settimana centrale del mese
    
    # Valori medi per initialstock e finalstock
    initialstock_mean = df[df['product_name'] == product_name]['initialstock'].mean()
    finalstock_mean = df[df['product_name'] == product_name]['finalstock'].mean()
    
    # Creazione della feature matrix
    X_new = np.array([[product_encoded, month, day_of_week, season, week_of_year, initialstock_mean, finalstock_mean]])
    
    # Normalizzazione
    X_new_scaled = scaler_X.transform(X_new)
    
    # Predizione
    y_pred_norm = model.predict(X_new_scaled).flatten()
    y_pred_denorm_log = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    
    # Invertire la trasformazione logaritmica
    predicted_qty = np.expm1(y_pred_denorm_log)
    
    # Controlla l'unità di misura
    uom = df[df['product_name'] == product_name]['uom'].iloc[0]
    if uom == "Numero" or pd.isna(uom): #Il file 2022 non presenta la colonna "uom" quindi viene salvato in df come "nan" 
                                        #e lo tratto come se fosse un Numero intero
        return np.ceil(predicted_qty).astype(int)  # Converte in intero arrotondando al numero superiore
    elif uom == "MT" or uom == "KG":
        return predicted_qty  # Ritorna come float
    else:
        raise ValueError(f"Unità di misura sconosciuta: {uom}")

# Esempio di previsione
product_name = 'MOLLA'
month = 5  # Maggio
predicted_qty = predict_quantity(product_name, month)
print(f"Quantità prevista per {product_name} nel mese {month}: {predicted_qty[0]}")
