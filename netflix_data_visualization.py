import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("netflix_data.csv")
df = df.drop_duplicates()

df["Country"] = df["Country"].fillna("Unknown")
df["Rating"] = df["Rating"].fillna("Not Rated")
df["Genre"] = df["Genre"].fillna("Unknown")
df["Director"] = df["Director"].fillna("Not Given")
df["Cast"] = df["Cast"].fillna("Not Given")

df["Date_Added"] = pd.to_datetime(df["Date_Added"], errors="coerce")
df["Duration"] = df["Duration"].astype(str)
df["Duration_Min"] = df["Duration"].str.extract(r"(\d+)").astype(float)

type_count = df["Type"].value_counts()
rating_count = df["Rating"].value_counts().head(8)
country_count = df["Country"].value_counts().head(6)
genre_count = df["Genre"].value_counts().head(6)
year_count = df["Release_Year"].value_counts().sort_index()
movie_tv_count = df.groupby("Release_Year")["Type"].value_counts().unstack().fillna(0)

plt.figure(figsize=(6,6))
plt.pie(
    type_count,
    labels=type_count.index,
    autopct="%1.1f%%",
    colors=["#ff6f61", "#6a5acd"],
    wedgeprops={"edgecolor": "black", "linewidth": 1.2}
)
plt.title("Netflix Content Type Distribution")
plt.show()

plt.figure(figsize=(8,5))
plt.bar(
    rating_count.index,
    rating_count.values,
    color="#2ecc71",
    edgecolor="black",
    linewidth=1.2
)
plt.title("Top Content Ratings on Netflix")
plt.show()

plt.figure(figsize=(8,5))
plt.bar(
    country_count.index,
    country_count.values,
    color="#3498db",
    edgecolor="black",
    linewidth=1.2
)
plt.title("Top Content Producing Countries")
plt.show()

plt.figure(figsize=(8,5))
plt.bar(
    genre_count.index,
    genre_count.values,
    color="#e74c3c",
    edgecolor="black",
    linewidth=1.2
)
plt.title("Most Popular Genres on Netflix")
plt.show()
