# 📈 Sales Prediction Project for Supply Chain Management (SCM) Optimization

Business Problem:

In supply chain management (SCM), accurate sales forecasting is critical for optimizing stock levels, reducing excess inventory, and minimizing stockouts. Traditional approaches often rely on historical trends or manual estimates, which are not adaptive to real-time market dynamics such as promotions, competitor pricing, and seasonal variations.

Why This Project Matters:
- 📉 Overstock or Understock can lead to loss in revenue or increased holding costs.
- ⚙️ Demand-Supply Mismatch leads to poor customer satisfaction.
- 💰 Promotion Planning & Pricing Strategy can be guided by accurate sales predictions.
- 🎯 Data-Driven Decisions lead to leaner operations and better forecasting.

This ML-powered sales prediction tool empowers business stakeholders with insights to better forecast store-level and product-level demand, enabling smarter procurement and logistics planning.

---

## 🏷️ Project Description

This project aims to build a machine learning-based sales prediction model to help optimize supply chain management. With accurate sales forecasting, companies can better manage inventory, reduce costs, and improve customer satisfaction through precise and efficient planning.

---

## 🚀 Key Features

- Sales prediction based on historical data and related factors.
- Automated data preprocessing including date feature extraction, missing value imputation, and categorical data encoding.
- Robust Random Forest Regressor model capable of handling numerical and categorical data.
- Model evaluation using R² metric to ensure prediction quality.
- Interactive Streamlit application for data upload, manual input, and visualization of prediction results.
- Graphical visualization to understand sales trends by store and product.

---

## 🔄 Workflow

### Business Workflow
[Raw Supply Chain Data] 

      ↓
      
[Feature Extraction: Date, Price, Stock, etc.]

      ↓
      
[Sales Prediction via Trained ML Model]

      ↓
      
[Insights: Store/Product Demand Forecasting]

      ↓
      
[Business Decisions: Stocking, Promotions, Logistics]


### Technical Workflow
1. Read CSV data and process date columns.
2. Preprocess data: impute missing values, scale numerical data, and encode categorical variables.
3. Split data into training and testing sets (80%:20%).
4. Build an integrated pipeline combining preprocessing and Random Forest Regressor model.
5. Train the model on training data.
6. Evaluate model performance using R² score on test data.
7. Perform predictions and display results via Streamlit application.

---

## 🧠 Machine Learning Model Explanation

The model used is a **Random Forest Regressor**, which is an ensemble of decision trees.

- **Advantages**:
  - Reduces risk of overfitting by aggregating many decision trees.
  - Handles non-linear relationships between features and target without complex parameter tuning.
  - Capable of handling numerical and categorical data after preprocessing.
  - Provides feature importance to understand dominant factors in sales.

- **How it works**:
  - Builds multiple decision trees on different data subsets
  - Averages outputs to improve accuracy and reduce overfitting
  - Suitable for **complex interactions** between features like price, promotion, stock level, etc.

---

## 🛠️ How to Use the Streamlit Application

### Data Upload
- Upload a CSV file containing these columns: Store, Product, Price, Promotion, Stock_Level, Competitor_Price, Date, and Sales.
- The data will be automatically processed, and the model will display sales predictions along with model performance.

### Manual Input
- Use the sidebar to manually enter feature values.
- The model instantly provides sales predictions based on the input values.

### Visualization
- Preview data with sales prediction columns.
- Bar charts showing average sales per store and product for visual analysis.

---

## 📋 Data Structure

| Column            | Data Type   | Description                          |
|-------------------|-------------|------------------------------------|
| Store             | Categorical | Store identifier                   |
| Product           | Categorical | Product type                      |
| Price             | Numerical   | Product price                     |
| Promotion         | Numerical   | Indicator of promotion (discount) |
| Stock_Level       | Numerical   | Product stock level               |
| Competitor_Price  | Numerical   | Competitor product price          |
| Date              | Date        | Transaction date                  |
| Sales             | Numerical   | Actual sales (target variable)   |

---

## 📊 Model Evaluation

| Metric   | Description                                         |
|----------|---------------------------------------------------|
| R² Score | Measures model prediction quality; values close to 1 indicate excellent prediction accuracy. |

---

## 🧩 5. Streamlit App Features

| Feature               | Description                                                              |
|-----------------------|--------------------------------------------------------------------------|
| 📤 File Uploader       | Upload a CSV dataset for batch predictions                              |
| 🧾 Manual Input        | Input features manually in the sidebar to get instant predictions        |
| 🧠 Auto Model Training | Automatically trains a Random Forest on the data                         |
| 📊 R² Score Display    | Shows model accuracy (goodness of fit)                                   |
| 📄 Data Preview        | Interactive table of actual and predicted sales                          |
| 📈 Sales Chart         | Bar chart showing average sales by Store and Product                     |
| 💼 Business-Ready UI   | Clean layout for non-technical users to get actionable insights          |

---

---

## Project Summary

This project demonstrates how machine learning can be effectively applied to **sales prediction within supply chain management**. By leveraging historical data and domain-specific features such as price, promotions, stock levels, and temporal patterns, the app empowers businesses to make data-driven decisions that improve operational efficiency and profitability.

With a user-friendly interface powered by Streamlit, it offers both **real-time manual predictions** and **batch processing via CSV uploads**, making it accessible to business users, analysts, and technical teams alike.

### ✅ Key Outcomes:
- 📈 Improved sales forecasting accuracy
- 📦 Better inventory planning and demand alignment
- 💰 Informed pricing and promotional strategies
- 🔍 Actionable insights through interactive dashboards and performance metrics

This project sets a strong foundation for **future enhancements** such as real-time API deployment, time-series modeling, and integration with enterprise ERP or SCM platforms. It bridges the gap between data science and business impact — turning raw data into reliable, interpretable forecasts.

---

Application Link

Click Here [link Application](https://scm-sales-prediction.streamlit.app/)

