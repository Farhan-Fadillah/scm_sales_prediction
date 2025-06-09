import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_default_data():
    df = pd.read_csv('supply_chain_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df.drop('Date', axis=1, inplace=True)
    return df

@st.cache_resource
def train_model(df):
    X = df.drop('Sales', axis=1)
    y = df['Sales']
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

    pipeline.numeric_features = numeric_features
    pipeline.categorical_features = categorical_features

    return pipeline, score

def predict_manual_input(model):
    st.sidebar.header("Input Data Manual untuk Prediksi")

    store = st.sidebar.selectbox("Store", options=['Store_A', 'Store_B', 'Store_C'])
    product = st.sidebar.selectbox("Product", options=['Product_1', 'Product_2', 'Product_3', 'Product_4'])
    price = st.sidebar.number_input("Price", min_value=0.0, value=10.0)
    promotion = st.sidebar.number_input("Promotion (0 atau 1)", min_value=0, max_value=1, value=0)
    stock_level = st.sidebar.number_input("Stock Level", min_value=0, value=100)
    competitor_price = st.sidebar.number_input("Competitor Price", min_value=0.0, value=9.5)
    month = st.sidebar.slider("Month (1-12)", 1, 12, 6)
    day_of_week = st.sidebar.slider("Day of Week (0=Senin,..6=Minggu)", 0, 6, 2)

    input_df = pd.DataFrame({
        'Store': [store],
        'Product': [product],
        'Price': [price],
        'Promotion': [promotion],
        'Stock_Level': [stock_level],
        'Competitor_Price': [competitor_price],
        'Month': [month],
        'DayOfWeek': [day_of_week]
    })

    expected_cols = model.numeric_features + model.categorical_features
    missing_cols = set(expected_cols) - set(input_df.columns)
    if missing_cols:
        st.error(f"Data input manual kurang kolom: {missing_cols}")
        return input_df, None

    input_df = input_df[expected_cols]
    pred = model.predict(input_df)[0]
    st.sidebar.markdown(f"### Prediksi Sales: {pred:.2f}")

    return input_df, pred

def main():
    st.title("📦 Supply Chain Optimization - Sales Prediction")
    st.markdown("""
    ### Cara Pakai Aplikasi:
    - **Upload CSV**: Format kolom harus termasuk: `Store, Product, Price, Promotion, Stock_Level, Competitor_Price, Date, Sales`
    - **Atau Input Manual** melalui sidebar untuk prediksi cepat.
    - Visualisasi akan muncul setelah prediksi.
    """)

    df_default = load_default_data()
    model, r2_score = train_model(df_default)

    st.subheader("Model Performance pada Dataset Default")
    st.write(f"📊 R² Score: **{r2_score:.2f}**")

    uploaded_file = st.file_uploader("📤 Upload file CSV data supply chain", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Hapus kolom Unnamed (biasanya hasil dari index CSV)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Month'] = df['Date'].dt.month
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df.drop('Date', axis=1, inplace=True)
        else:
            st.error("⚠️ File harus mengandung kolom 'Date'.")
            return

        X_input = df.drop(columns=['Sales'], errors='ignore')
        y_true = df['Sales'] if 'Sales' in df.columns else None

        expected_cols = model.numeric_features + model.categorical_features
        missing_cols = set(expected_cols) - set(X_input.columns)
        if missing_cols:
            st.error(f"⚠️ Data upload kurang kolom: {missing_cols}")
        else:
            X_input = X_input[expected_cols]
            y_pred = model.predict(X_input)
            df['Predicted_Sales'] = y_pred

            st.subheader("📄 Preview Data dengan Prediksi Sales")
            st.dataframe(df.head(20))

            if y_true is not None:
                from sklearn.metrics import r2_score as r2_metric
                r2_uploaded = r2_metric(y_true, y_pred)
                st.write(f"🔍 R² Score pada data upload: **{r2_uploaded:.2f}**")

            grouped = df.groupby(['Store', 'Product'])['Predicted_Sales'].mean().reset_index()
            st.subheader("📊 Rata-rata Prediksi Sales per Store dan Product")
            st.dataframe(grouped)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=grouped, x='Store', y='Predicted_Sales', hue='Product', ax=ax)
            ax.set_title('Rata-rata Prediksi Sales per Store dan Product')
            ax.set_ylabel('Predicted Sales')
            st.pyplot(fig)
    else:
        st.info("📥 Upload file CSV atau gunakan input manual di sidebar.")

    input_df, manual_pred = predict_manual_input(model)
    st.subheader("Input Manual dan Prediksi")
    st.write("Data input manual:")
    st.table(input_df)

    if manual_pred is not None:
        st.success(f"Prediksi sales dari input manual: **{manual_pred:.2f}**")
    else:
        st.warning("Prediksi tidak tersedia karena input kurang lengkap.")

if __name__ == "__main__":
    main()

st.markdown("---")
st.markdown("Made by Farhan Fadillah")