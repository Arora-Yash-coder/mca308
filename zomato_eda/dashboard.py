import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Zomato EDA & ML Dashboard", layout="wide")

# -----------------------------------------------------------------------------
# MODEL + ENCODER LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("rating_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        enc = pickle.load(f)
    return model, enc

model, enc = load_model()

# -----------------------------------------------------------------------------
# SAFE ENCODING WRAPPER
# -----------------------------------------------------------------------------
def apply_encoding(enc, input_df, cols):
    for col in cols:
        input_df[col] = enc[col].transform(input_df[col].astype(str))
    return input_df

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("zomato.csv")

    df["rate"] = (
        df["rate"].astype(str)
        .str.replace("/5", "")
        .replace(["NEW", "-", "nan"], np.nan)
    )
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")

    df["approx_cost(for two people)"] = (
        df["approx_cost(for two people)"].astype(str).str.replace(",", "")
    )
    df["approx_cost(for two people)"] = pd.to_numeric(
        df["approx_cost(for two people)"], errors="coerce"
    )

    df["rest_type"] = df["rest_type"].fillna("Unknown")
    df["listed_in(city)"] = df["listed_in(city)"].fillna("Unknown")
    df["listed_in(type)"] = df["listed_in(type)"].fillna("Unknown")

    return df

df = load_data()

# -----------------------------------------------------------------------------
# CACHED WORDCLOUD (dish liked only)
# -----------------------------------------------------------------------------
@st.cache_data
def generate_cloud(text):
    return WordCloud(width=1600, height=900, background_color="white").generate(text)

# -----------------------------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")

city = st.sidebar.multiselect("City", sorted(df["listed_in(city)"].unique()))
rest_type = st.sidebar.multiselect("Restaurant Type", sorted(df["rest_type"].unique()))
cuisine_filter = st.sidebar.text_input("Cuisine Keyword")

online = st.sidebar.selectbox("Online Order", ["All", "Yes", "No"])
book = st.sidebar.selectbox("Book Table", ["All", "Yes", "No"])

filtered = df.copy()
if city:
    filtered = filtered[filtered["listed_in(city)"].isin(city)]
if rest_type:
    filtered = filtered[filtered["rest_type"].isin(rest_type)]
if cuisine_filter:
    filtered = filtered[
        filtered["cuisines"].str.contains(cuisine_filter, case=False, na=False)
    ]
if online != "All":
    filtered = filtered[filtered["online_order"] == online]
if book != "All":
    filtered = filtered[filtered["book_table"] == book]

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Overview", "Ratings", "WordClouds", "Radar Chart",
        "Price & Votes", "Correlation", "Prediction", "Findings"
    ]
)

# -----------------------------------------------------------------------------
# TAB 1 — OVERVIEW
# -----------------------------------------------------------------------------
with tab1:
    st.title("🍽️ Zomato EDA & Machine Learning Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Restaurants", len(filtered))
    c2.metric("Avg Rating", round(filtered["rate"].mean(), 2))
    c3.metric("Avg Cost for Two", round(filtered["approx_cost(for two people)"].mean(), 2))

    fig = px.histogram(
        filtered,
        x="rate",
        nbins=20,
        title="Rating Distribution",
        color_discrete_sequence=["#ff4b4b"],
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2 — RATINGS
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Ratings by Restaurant Type")
    fig = px.box(filtered, x="rest_type", y="rate", color="rest_type")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Online Order vs Rating")
    fig = px.box(filtered, x="online_order", y="rate", color="online_order")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3 — WORDCLOUDS (Dish Liked Only)
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Dish Liked WordCloud")

    text = " ".join(filtered["dish_liked"].dropna().astype(str))
    if text.strip():
        wc = generate_cloud(text)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# TAB 4 — RADAR CHART
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Radar Chart: Aggregated Metrics")

    metrics = {
        "Average Rating": filtered["rate"].mean(),
        "Average Votes": filtered["votes"].mean(),
        "Average Cost": filtered["approx_cost(for two people)"].mean(),
        "Online Order %": filtered["online_order"].eq("Yes").mean() * 5,
        "Book Table %": filtered["book_table"].eq("Yes").mean() * 5,
    }

    categories = list(metrics.keys())
    values = list(metrics.values())
    values.append(values[0])

    fig = go.Figure(data=[go.Scatterpolar(r=values, theta=categories, fill="toself")])
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)))
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5 — PRICE & VOTES
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("Cost Distribution")
    fig = px.histogram(filtered, x="approx_cost(for two people)", nbins=30)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Votes vs Rating (with Trendline)")
    fig = px.scatter(
        filtered, x="votes", y="rate",
        trendline="ols", color="rate"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 6 — CORRELATION
# -----------------------------------------------------------------------------
with tab6:
    num_df = filtered.select_dtypes(include=[np.number])
    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# TAB 7 — PREDICTION (NO TAB REFRESH)
# -----------------------------------------------------------------------------
with tab7:
    st.header("Predict Restaurant Rating Category")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)

        votes = c1.number_input("Votes", 0, 100000, 100)
        cost = c1.number_input("Cost for Two (same as dataset)", 50, 10000, 500)

        online_input = c2.selectbox("Online Order", ["Yes", "No"])
        book_input = c2.selectbox("Book Table", ["Yes", "No"])

        # Correct columns from the dataset
        type_input = c1.selectbox("Listing Type", sorted(df["listed_in(type)"].unique()))
        city_input = c2.selectbox("City", sorted(df["listed_in(city)"].unique()))
        rest_type_input = c2.selectbox("Restaurant Type", sorted(df["rest_type"].unique()))

        submitted = st.form_submit_button("Predict Rating")

    if submitted:
        # EXACT SAME ORDER AS MODEL TRAINING
        input_df = pd.DataFrame([{
            "votes": votes,
            "cost": cost,
            "online_order": 1 if online_input == "Yes" else 0,
            "book_table": 1 if book_input == "Yes" else 0,
            "type": type_input,
            "city": city_input,
            "rest_type": rest_type_input
        }])

        # Apply encoders saved earlier
        categorical_cols = ["type", "city", "rest_type"]
        input_df = apply_encoding(enc, input_df, categorical_cols)

        pred = model.predict(input_df)[0]
        st.success(f"⭐ Predicted Rating Category: {pred}")


# -----------------------------------------------------------------------------
# TAB 8 — FINDINGS
# -----------------------------------------------------------------------------
with tab8:
    st.header("Findings & Insights")
    st.markdown("""
### Key Insights
1. Most restaurants fall between 3.2–4.3 rating.
2. Restaurants with online ordering tend to have slightly higher ratings.
3. Mid-range cost restaurants (₹300–800) show highest popularity.
4. Votes correlate strongly with customer engagement.
5. Dish-liked wordcloud reveals popular items across the city.
6. Random Forest predicts ratings based on votes, cost, online order, booking, restaurant type, listing type, and city.
""")

    st.text_area("Additional Notes")
