#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


##1. Clean and Prepare Feedback Data


# In[2]:


df = pd.read_csv("student_feedback.csv")


# In[3]:


df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r"[^\w\s]", "", regex=True)


# In[4]:


df = df.drop(columns=["unnamed_0"], errors='ignore')


# In[5]:


df.head()


# In[ ]:


##2. Analyze Ratings (1–5 or 1–10 scale)


# In[6]:


rating_cols = df.drop(columns=["student_id"], errors='ignore').select_dtypes(include='number').columns


# In[7]:


average_ratings = df[rating_cols].mean().sort_values(ascending=False)
average_ratings


# In[ ]:


##3. Use NLP Tools to Score Sentiment in Comments


# In[10]:


import random

sample_comments = [
    "Great session, really helpful!",
    "Too fast, couldn’t follow.",
    "Excellent and well-organized.",
    "It was okay, but could be better.",
    "Not engaging at all.",
    "Loved the content!",
    "Difficult to understand",
    "Very informative and to the point.",
    "More examples would help.",
    "Could improve presentation style."
]

df["comments"] = [random.choice(sample_comments) for _ in range(len(df))]


# In[11]:


def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["comments"].astype(str).apply(get_sentiment)
df["sentiment"].value_counts()


# In[ ]:


## 4. Visualize Trends with Beautiful Charts and Graphs


# In[12]:


plt.figure(figsize=(10, 5))
sns.barplot(x=average_ratings.values, y=average_ratings.index, palette="viridis")
plt.title("Average Ratings by Category")
plt.xlabel("Average Score")
plt.ylabel("Question")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[13]:


sentiment_counts = df["sentiment"].value_counts()
plt.figure(figsize=(5, 5))
sentiment_counts.plot.pie(autopct='%1.1f%%', colors=['#8BC34A', '#FFC107', '#F44336'])
plt.title("Sentiment Breakdown")
plt.ylabel("")
plt.show()


# In[ ]:


##5. Suggest Improvements for Future Events


# In[14]:


recommendations = []
for category, score in average_ratings.items():
    readable = category.replace("_", " ").capitalize()
    if score < 6:
        recommendations.append(f"⚠️ Improve **{readable}** (avg: {score:.2f})")
    elif score >= 8:
        recommendations.append(f"✅ Continue excelling in **{readable}** (avg: {score:.2f})")


# In[15]:


print("### 📌 Key Recommendations\n")
for line in recommendations:
    print(line)


# In[ ]:




