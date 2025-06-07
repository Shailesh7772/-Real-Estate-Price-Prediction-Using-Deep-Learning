# 🏠 Real Estate Price Prediction Using Deep Learning

> **IEEE Publication Under Review**  
> A deep learning-based comparative analysis for time-series house price forecasting using Zillow dataset.

---

## 📘 Project Overview

This research investigates the application of state-of-the-art **deep learning models** to predict **real estate prices** by analyzing temporal patterns in housing data. Traditional statistical models often fall short in capturing **non-linear dependencies**, which deep learning models overcome with higher accuracy and better generalization.

---

## 🗃️ Dataset Summary

- 📊 **Source**: [Zillow Housing Dataset](https://www.zillow.com/research/data/)
- 🏘️ **Rows**: 895
- 📍 **Columns**: 305 (regions, home values, dates)
- 📅 **Time Span**: January 2000 – December 2024
- 🧹 **Preprocessing**: Wide-to-long format conversion using `pandas.melt`, null value handling, scaling

---

## 🧠 Deep Learning Models Evaluated

| Model          | Type                        | Key Feature                            |
|----------------|-----------------------------|----------------------------------------|
| **NeuTS**      | LSTM-based                  | Sequential pattern extraction          |
| **TimeGPT**    | Transformer-based (pretrained) | Zero-shot, long-term forecasting     |
| **TCN**        | Temporal Convolutional Net  | Causal & dilated convolutions          |
| **CNN-LSTM**   | Hybrid CNN + LSTM           | Spatial-temporal feature fusion        |
| **DeepAR**     | Autoregressive LSTM         | Probabilistic time-series forecasting  |

---

## 🧪 Evaluation Metrics

We used three key metrics to evaluate prediction accuracy:

- **MAE (Mean Absolute Error)**: Measures average magnitude of errors  
- **MSE (Mean Squared Error)**: Penalizes larger errors more strongly  
- **R² Score (Coefficient of Determination)**: Measures goodness of fit  

---

## 📊 Performance Comparison (Test Set)

| Model       | MAE       | MSE       | R² Score |
|-------------|-----------|-----------|----------|
| NeuTS       | 0.0214    | 0.0007    | 0.9849   |
| TimeGPT     | 0.0317    | 0.0018    | 0.9960   |
| TCN         | 0.0084    | 0.0026    | 0.9975   |
| CNN-LSTM    | 0.0106    | 0.0173    | 0.9995   |
| DeepAR      | 0.0178    | 0.0033    | 0.9968   |

### ✅ Observations:

- **CNN-LSTM** has the **best R² score** (0.9995)
- **TCN** shows the **lowest MAE** (0.0084)
- **NeuTS** provides **low MSE** but slightly lower R²

---

## 🖼️ Forecast Visualizations

We forecasted housing prices in **New York, NY** using top-performing models.

### 🔹 NeuTS
- Accurate short-term predictions
- Smooth transitions in price curves

### 🔹 TCN
- Excellent pattern tracking
- Best MAE value for sharp price changes

### 🔹 TimeGPT
- Strong long-term forecasting
- Wider prediction intervals

### 🔹 Streamlit Frontend
- Web-based UI to select cities and visualize trends interactively

> _Note: Screenshots or demo GIFs can be embedded here if desired._

---

## 🧰 Tech Stack

- 🧪 Models: NeuTS, TimeGPT, TCN, CNN-LSTM, DeepAR (PyTorch, Keras)
- 📊 Visualization: Plotly, Matplotlib
- 🌐 Frontend: Streamlit
- 📦 Data Handling: Pandas, NumPy
- 🛠️ Tools: Jupyter, VS Code

---

## 🚀 Web Application (Streamlit)

Run the interactive dashboard locally:

```bash
pip install -r requirements.txt
streamlit run app.py
