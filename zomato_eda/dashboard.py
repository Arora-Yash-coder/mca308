import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go

st.set_page_config(page_title="Zomato EDA & ML Dashboard", layout="wide")

# -----------------------------------------------------------------------------
# Load Model and Encoders
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
# Load Data
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("zomato.csv")

    df["rate"] = (
        df["rate"]
        .astype(str)
        .str.replace("/5", "")
        .replace(["NEW", "-", "nan"], np.nan)
    )
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")

    df["approx_cost(for two people)"] = (
        df["approx_cost(for two people)"]
        .astype(str)
        .str.replace(",", "")
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
# Cached WordCloud Generator
# -----------------------------------------------------------------------------
@st.cache_data
def generate_cloud(text):
    return WordCloud(width=1600, height=900, background_color="white").generate(text)

# -----------------------------------------------------------------------------
# Sidebar Filters
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")

city = st.sidebar.multiselect("City", sorted(df["listed_in(city)"].dropna().unique()))
rest_type = st.sidebar.multiselect("Restaurant Type", sorted(df["rest_type"].dropna().unique()))
cuisine_filter = st.sidebar.text_input("Cuisine Keyword")

online = st.sidebar.selectbox("Online Order", ["All", "Yes", "No"])
book = st.sidebar.selectbox("Book Table", ["All", "Yes", "No"])

filtered = df.copy()
if city:
    filtered = filtered[filtered["listed_in(city)"].isin(city)]
if rest_type:
    filtered = filtered[filtered["rest_type"].isin(rest_type)]
if cuisine_filter:
    filtered = filtered[filtered["cuisines"].str.contains(cuisine_filter, case=False, na=False)]
if online != "All":
    filtered = filtered[filtered["online_order"] == online]
if book != "All":
    filtered = filtered[filtered["book_table"] == book]

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["Overview", "Ratings", "WordClouds", "Radar Chart", "Price & Votes",
     "Correlation", "Prediction", "Findings"]
)

# -----------------------------------------------------------------------------
# TAB 1: Overview
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
        color_discrete_sequence=["#ff4b4b"]
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: Ratings
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Ratings by Restaurant Type")
    fig = px.box(filtered, x="rest_type", y="rate", color="rest_type")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Online Order vs Rating")
    fig = px.box(filtered, x="online_order", y="rate", color="online_order")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: WordClouds (Dish Liked ONLY — Reviews Removed)
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("WordClouds")

    col1, _ = st.columns(2)

    with col1:
        st.write("**Dish Liked WordCloud**")
        text1 = " ".join(filtered["dish_liked"].dropna().astype(str))
        if text1.strip():
            wc = generate_cloud(text1)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)

# -----------------------------------------------------------------------------
# TAB 4: Radar Chart
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Radar Chart: Aggregated Metrics")

    metrics = {
        "Average Rating": filtered["rate"].mean(),
        "Average Votes": filtered["votes"].mean(),
        "Average Cost": filtered["approx_cost(for two people)"].mean(),
        "Online Order %": (filtered["online_order"].eq("Yes").mean() * 5),
        "Book Table %": (filtered["book_table"].eq("Yes").mean() * 5),
    }

    categories = list(metrics.keys())
    values = list(metrics.values())
    values.append(values[0])

    fig = go.Figure(
        data=[go.Scatterpolar(r=values, theta=categories, fill="toself")]
    )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)))
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: Price & Votes
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("Cost Distribution")
    fig = px.histogram(filtered, x="approx_cost(for two people)", nbins=30)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Votes vs Rating (Trendline Enabled)")
    fig = px.scatter(
        filtered,
        x="votes",
        y="rate",
        trendline="ols",
        color="rate"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 6: Correlation Heatmap
# -----------------------------------------------------------------------------
with tab6:
    num_df = filtered.select_dtypes(include=[np.number])
    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# TAB 7: Prediction
# -----------------------------------------------------------------------------
with tab7:
    st.header("Predict Restaurant Rating Category")

    c1, c2 = st.columns(2)

    votes = c1.number_input("Votes", 0, 100000, 100)
    cost = c1.number_input("Cost for Two", 50, 10000, 500)

    online_input = c2.selectbox("Online Order", ["Yes", "No"])
    book_input = c2.selectbox("Book Table", ["Yes", "No"])

    rest_type_input = c1.selectbox("Restaurant Type", sorted(df["rest_type"].unique()))
    city_input = c2.selectbox("City", sorted(df["listed_in(city)"].unique()))
    type_input = c2.selectbox("Listing Type", sorted(df["listed_in(type)"].unique()))

    if st.button("Predict Rating"):
        input_df = pd.DataFrame([{
            "votes": votes,
            "approx_cost(for two people)": cost,
            "online_order": 1 if online_input == "Yes" else 0,
            "book_table": 1 if book_input == "Yes" else 0,
            "rest_type": rest_type_input,
            "listed_in(city)": city_input,
            "listed_in(type)": type_input
        }])

        for col in ["rest_type", "listed_in(city)", "listed_in(type)"]:
            input_df[col] = enc[col].transform(input_df[col].astype(str))

        pred = model.predict(input_df)[0]
        st.success(f"⭐ Predicted Rating Category: {pred}")

# -----------------------------------------------------------------------------
# TAB 8: Findings
# -----------------------------------------------------------------------------
with tab8:
    st.header("Findings & Insights")

    st.markdown("""
### Key Insights

1. Most restaurants fall between 3.2–4.3 rating.
2. Restaurants with online ordering tend to have slightly higher ratings.
3. Mid-price restaurants (₹300–800) perform best.
4. Votes correlate strongly with popularity.
5. WordCloud analysis shows popular dishes like biryani, paneer, tikka, chicken.
6. Random Forest predicts based on cost, votes, online order, table booking, restaurant type, listing type, and city.
""")

    notes = st.text_area("Add additional observations:")
    if notes:
        st.info("Notes saved (not persistent).")
