# 📦 Warehouse Forecast Application

Welcome to **Warehouse Forecast Application**! 🚀
This application predicts the required quantity of a product for a specific month in the warehouse. It uses historical data and a **Machine Learning** model to generate accurate forecasts and efficiently support inventory management. 📊

---

## ✨ Features
✅ **User-friendly interface** built with **Streamlit**  
✅ **Stock forecasting** based on product name and selected month  
✅ **Clear and interactive visualization** of results  
✅ **Online version available** for easy access  

---

## 🌍 Try the Online Application
Access the live version of the app directly from here:  
👉 **[Warehouse Forecast Application](https://zecaproject-warehouseforecast.streamlit.app/)**

---

## 🚀 How to Run Locally

### 📌 Prerequisites
Make sure you have installed:
- **Python 3.9 or higher** 🐍
- **pip** (Python package manager)

### 🛠 Installation
1️⃣ **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/zecaproject.git
   cd zecaproject
   ```

2️⃣ **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3️⃣ **Run the application**:
   ```bash
   streamlit run app.py
   ```

4️⃣ **Access the app** in your browser:
   ```
   http://localhost:8501
   ```

---

## 📂 Project Structure
📝 **`app.py`** - Main Streamlit application file  
🧠 **`model.h5`** - Pre-trained Machine Learning model  
📊 **`scaler_X.pkl` and `scaler_y.pkl`** - Saved scalers for feature normalization  
📦 **`requirements.txt`** - List of dependencies  
🖼 **`logo_zeca_ita.jpg`** - Company logo for the user interface  

---

## 📊 Data Structure (Confidential)
For confidentiality reasons, the raw data files cannot be shared. However, the dataset contains warehouse stock movement information structured as follows:

- **`product_name`**: Name of the product.
- **`initialstock`**: Initial quantity of the product in stock.
- **`finalstock`**: Final quantity of the product in stock.
- **`movementdate`**: Date of product movement.
- **`qty`**: Quantity moved.
- **`uom`**: Unit of measure (e.g., pieces, meters, kg).
- **`warehouse`**: Warehouse reference.
- **`qtyload`** and **`qtyunload`**: Quantity loaded/unloaded.

These structured files are used to train and make predictions within the application.

---

## 🔧 How to Use
1️⃣ **Enter the product name** in the search box 🔍  
2️⃣ **Select the product** from the dropdown list 📋  
3️⃣ **Choose the month** for the forecast 📅  
4️⃣ **Click "Predict"** to get the predicted quantity 🎯  

---

## 🛠 Troubleshooting
❌ **Error: `streamlit` command not found**
- Ensure Python is added to your system's `PATH` environment variable.
- Check Python and pip versions:
  ```bash
  python --version
  pip --version
  ```

❌ **Dependencies not installed?**
- Run:
  ```bash
  pip install -r requirements.txt
  ```

❌ **Issues with file paths?**
- Make sure all required files (`model.h5`, `scaler_X.pkl`, `scaler_y.pkl`, etc.) are in the same directory as `app.py`.

---

## 🎉 Enjoy!
Have fun using **Warehouse Forecast Application** and improve warehouse management smartly! 🤖📦

