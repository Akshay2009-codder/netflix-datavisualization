import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("netflix_data.csv")
df = df.drop_duplicates()

# -----------------------------
# 2. Data Cleaning
# -----------------------------
df["Country"] = df["Country"].fillna("Unknown")
df["Rating"] = df["Rating"].fillna("Not Rated")
df["Genre"] = df["Genre"].fillna("Unknown")
df["Director"] = df["Director"].fillna("Not Given")
df["Cast"] = df["Cast"].fillna("Not Given")

df["Date_Added"] = pd.to_datetime(df["Date_Added"], errors="coerce")

df["Duration"] = df["Duration"].astype(str)
df["Duration_Min"] = df["Duration"].str.extract(r"(\d+)").astype(float)

# -----------------------------
# 3. NumPy Operations
# -----------------------------
release_year_np = df["Release_Year"].dropna().to_numpy()
duration_np = df["Duration_Min"].dropna().to_numpy()

print("----- NumPy Statistics -----")
print("Mean Release Year:", np.mean(release_year_np))
print("Median Release Year:", np.median(release_year_np))
print("Standard Deviation:", np.std(release_year_np))
print("Minimum Release Year:", np.min(release_year_np))
print("Maximum Release Year:", np.max(release_year_np))

print("Average Duration:", np.mean(duration_np))
print("Max Duration:", np.max(duration_np))
print("Min Duration:", np.min(duration_np))
print("-----------------------------")

# -----------------------------
# 4. Seaborn Styling
# -----------------------------
sns.set_style("whitegrid")

# -----------------------------
# 5. Content Type Distribution
# -----------------------------
plt.figure(figsize=(6,5))
sns.countplot(data=df, x="Type", palette="Set2")
plt.title("Netflix Content Type Distribution")
plt.xlabel("Type")
plt.ylabel("Count")
plt.show()

# -----------------------------
# 6. Top 8 Ratings
# -----------------------------
plt.figure(figsize=(8,5))
top_ratings = df["Rating"].value_counts().head(8)
sns.barplot(x=top_ratings.index, y=top_ratings.values, palette="viridis")
plt.title("Top 8 Content Ratings")
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# 7. Top 6 Countries
# -----------------------------
plt.figure(figsize=(8,5))
top_countries = df["Country"].value_counts().head(6)
sns.barplot(x=top_countries.index, y=top_countries.values, palette="coolwarm")
plt.title("Top 6 Content Producing Countries")
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# 8. Top 6 Genres
# -----------------------------
plt.figure(figsize=(8,5))
top_genres = df["Genre"].value_counts().head(6)
sns.barplot(x=top_genres.index, y=top_genres.values, palette="magma")
plt.title("Most Popular Genres on Netflix")
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# 9. Content Growth Over Years
# -----------------------------
plt.figure(figsize=(10,6))
year_count = df["Release_Year"].value_counts().sort_index()
sns.lineplot(x=year_count.index, y=year_count.values)
plt.title("Netflix Content Growth Over Years")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")
plt.show()

# -----------------------------
# 10. Duration Distribution
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["Duration_Min"], bins=20, kde=True)
plt.title("Distribution of Content Duration")
plt.xlabel("Duration (Minutes)")
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# 11. Heatmap (Movies vs TV Shows by Year)
# -----------------------------
plt.figure(figsize=(12,6))
movie_tv_count = df.groupby("Release_Year")["Type"].value_counts().unstack().fillna(0)
sns.heatmap(movie_tv_count, cmap="YlGnBu")
plt.title("Movies vs TV Shows by Release Year")
plt.show()
