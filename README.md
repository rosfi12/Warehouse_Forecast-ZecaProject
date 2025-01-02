# Warehouse Forecast Application

This application predicts the required quantity of a product for a given month in the warehouse. It uses historical data and a pre-trained machine learning model to generate accurate predictions.

## Features
- User-friendly interface built with **Streamlit**.
- Predicts the required quantity of a product based on its name and selected month.
- Displays predictions in a visually appealing format.
- Hosted version available for easy access.

## Hosted Application
You can access the live application directly via this URL:

👉 **[Warehouse Forecast Application](https://zecaproject-warehouseforecast.streamlit.app/)**

---

## How to Run Locally

### Prerequisites
Ensure you have the following installed on your system:
- **Python 3.9 or higher**
- **pip** (Python package manager)

### Installation Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/zecaproject.git
   cd zecaproject
   ```

2. **Install the required dependencies**:
   All required dependencies are listed in `requirements.txt`. Install them using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   Execute the following command to start the app locally:
   ```bash
   streamlit run app.py
   ```

4. **Access the application**:
   Open your browser and go to:
   ```
   http://localhost:8501
   ```

---

## Project Structure
- **`app.py`**: Main application file for Streamlit.
- **`model.h5`**: Pre-trained machine learning model for predictions.
- **`scaler_X.pkl` and `scaler_y.pkl`**: Saved scalers for feature normalization.
- **`requirements.txt`**: List of all dependencies required to run the application.
- **`logo_zeca_ita.jpg`**: Company logo displayed in the UI.

---

## Usage
1. **Enter the product name** in the search box.
2. **Select the desired product** from the dropdown.
3. **Choose the month** for the forecast.
4. **Click Predict** to see the forecasted quantity.

---

## Troubleshooting
- **Error: `streamlit` command not found**:
  - Ensure you have added Python to your system's PATH.
  - You can test this by running:
    ```bash
    python --version
    pip --version
    ```

- **Dependencies not installed**:
  - Make sure to install the dependencies using:
    ```bash
    pip install -r requirements.txt
    ```

- **Issues with file paths**:
  - Ensure all required files (`model.h5`, `scaler_X.pkl`, `scaler_y.pkl`, etc.) are in the same directory as `app.py`.

---

Enjoy using the **Warehouse Forecast Application**! 🎉
