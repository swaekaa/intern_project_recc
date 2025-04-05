import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import re

# Load the processed dataset
df = pd.read_csv("processed_data.csv")
df['Skills_List'] = df['Skills'].fillna("").apply(lambda x: [s.strip().lower() for s in x.split(',')])

# Convert Experience from string to numeric using a regex-based function
def convert_experience(exp_str):
    if pd.isna(exp_str):
        return 0
    exp_str = exp_str.lower()
    if "+" in exp_str:
        return float(re.findall(r"\d+", exp_str)[0]) + 1
    numbers = list(map(int, re.findall(r"\d+", exp_str)))
    if len(numbers) == 2:
        return sum(numbers) / 2
    elif len(numbers) == 1:
        return numbers[0]
    return 0

df['Experience'] = df['Experience'].apply(convert_experience)

# Vectorizers
vectorizer = TfidfVectorizer()
title_tfidf = vectorizer.fit_transform(df['Title'])

mlb = MultiLabelBinarizer()
skills_encoded = mlb.fit_transform(df['Skills_List'])

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
location_encoded = pd.DataFrame(encoder.fit_transform(df[['Location']]))

# Scale experience
scaler = StandardScaler()
experience_scaled = scaler.fit_transform(df[['Experience']])

# Combine all features
combined_features = hstack([title_tfidf, skills_encoded, location_encoded.values, experience_scaled])

# Streamlit UI
st.title("Internship Recommendation System")

user_title = st.text_input("Preferred Role (e.g. Data Analyst)")
user_skills = st.text_input("Your Skills (comma-separated, e.g. Python, SQL, Machine Learning)")
user_location = st.selectbox("Preferred Location", df['Location'].unique())
user_experience = st.slider("Years of Experience", 0, 10, 1)

if st.button("Recommend Internships"):
    # Process user input
    user_title_vec = vectorizer.transform([user_title])

    # Clean and match user input skills
    skills_list = [s.strip().lower() for s in user_skills.split(",") if s.strip()]
    filtered_skills = [s for s in skills_list if s in [item for sublist in df['Skills_List'].tolist() for item in sublist]]

    # Debugging
    print("Dataset skills:", [item for sublist in df['Skills_List'].tolist() for item in sublist])
    print("User filtered skills:", filtered_skills)

    if filtered_skills:
        # Calculate skill match score for each internship
        df['skill_match'] = df['Skills_List'].apply(
            lambda x: len(set(filtered_skills).intersection(set(x))) / len(filtered_skills) if filtered_skills else 0.5
        )
    else:
        df['skill_match'] = 0.5  # Neutral score for no skill matches

    # Location vector
    user_location_vec = pd.DataFrame(encoder.transform([[user_location]]))

    # Experience
    user_exp_scaled = scaler.transform([[user_experience]])

    # Combine user input into vector
    user_skills_vec = np.zeros((1, len(mlb.classes_)))
    for skill in filtered_skills:
        if skill in mlb.classes_:
            user_skills_vec[0, mlb.classes_.tolist().index(skill)] = 1

    user_combined = hstack([user_title_vec, user_skills_vec, user_location_vec.values, user_exp_scaled])

    # Compute similarity
    similarities = cosine_similarity(user_combined, combined_features).flatten()

    # Compute custom ranking score
    df['similarity'] = similarities
    df['location_match'] = df['Location'].apply(lambda x: 1 if x == user_location else 0)
    df['exp_diff'] = abs(df['Experience'] - user_experience)
    df['exp_score'] = df['exp_diff'].apply(lambda x: max(0, 1 - x / 10))  # normalize to [0,1]

    # Final Score
    df['score'] = 0.5 * df['skill_match'] + 0.2 * df['location_match'] + 0.1 * df['exp_score'] + 0.2 * df['similarity']

    top_recommendations = df.sort_values(by='score', ascending=False).head(10)

    # Display Results
    st.subheader("Top Internship Matches:")
    for i, row in top_recommendations.iterrows():
        st.markdown(f"**{row['Title']}** at *{row['Company']}*")
        st.markdown(f"📍 Location: {row['Location']} | 💼 Experience: {row['Experience']} years")
        st.markdown(f"🛠️ Skills: {', '.join(row['Skills_List'])}")
        st.markdown(f"🔗 [Apply Here]({row['Link']})")
        st.markdown(f"🎯 Match Score: {row['score']:.2f} | 🧠 Skill Match: {row['skill_match']:.2f}")
        st.markdown("---")
