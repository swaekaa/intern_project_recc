import streamlit as st
import pandas as pd
import numpy as np
import re
import fitz  # PyMuPDF
from streamlit_tags import st_tags
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# Experience conversion
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

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("processed_data.csv")
    df['Skills_List'] = df['Skills'].fillna("").apply(lambda x: [s.strip().lower() for s in x.split(',')])
    df['Experience'] = df['Experience'].apply(convert_experience)
    return df

df = load_data().head(300)  # Limit rows for faster prototyping

# Load smaller BERT model
@st.cache_resource
def load_bert_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


bert_model = load_bert_model()

@st.cache_data
def get_title_embeddings(df):
    return bert_model.encode(df['Title'].tolist(), show_progress_bar=True)

title_embeddings = get_title_embeddings(df)

# Other encoders
mlb = MultiLabelBinarizer()
skills_encoded = mlb.fit_transform(df['Skills_List'])

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
location_encoded = pd.DataFrame(encoder.fit_transform(df[['Location']]))

scaler = StandardScaler()
experience_scaled = scaler.fit_transform(df[['Experience']])

# Resume text extraction
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_title_and_skills(resume_text):
    title_keywords = ['data analyst', 'data scientist', 'web developer', 'machine learning', 'ai', 'android developer']
    skills_keywords = [skill.lower() for skill in mlb.classes_]

    title_found = ""
    for title in title_keywords:
        if title in resume_text.lower():
            title_found = title.title()
            break

    skills_found = list(set([word for word in resume_text.lower().split() if word in skills_keywords]))
    return title_found, skills_found

# Streamlit UI
st.title("🌟 Optimized Internship Recommender (Fast Mode)")

uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
resume_text = ""
auto_title = ""
auto_skills = []

if uploaded_file:
    with st.spinner("Extracting resume info..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        auto_title, auto_skills = extract_title_and_skills(resume_text)
    st.success("Resume parsed!")

user_title = st.text_input("Preferred Role", value=auto_title)
all_skill_suggestions = sorted(list(set(skill for sublist in df['Skills_List'] for skill in sublist)))
user_skills = st_tags(
    label='Your Skills',
    value=auto_skills,
    suggestions=all_skill_suggestions,
    maxtags=15,
    key='skills'
)
user_location = st.selectbox("Preferred Location", sorted(df['Location'].dropna().unique()))
user_experience = st.slider("Years of Experience", 0, 10, 1)

if st.button("🔍 Recommend Internships"):
    user_title_vec = bert_model.encode([user_title])
    title_similarities = cosine_similarity(user_title_vec.reshape(1, -1), title_embeddings).flatten()

    filtered_skills = [s.strip().lower() for s in user_skills if s.strip() in mlb.classes_]
    df['skill_match'] = df['Skills_List'].apply(
        lambda x: len(set(filtered_skills).intersection(set(x))) / len(filtered_skills) if filtered_skills else 0.5
    )

    user_location_vec = pd.DataFrame(encoder.transform([[user_location]]))
    user_exp_scaled = scaler.transform([[user_experience]])

    user_skills_vec = np.zeros((1, len(mlb.classes_)))
    for skill in filtered_skills:
        user_skills_vec[0, mlb.classes_.tolist().index(skill)] = 1

    df['similarity'] = title_similarities
    df['location_match'] = df['Location'].apply(lambda x: 1 if x == user_location else 0)
    df['exp_diff'] = abs(df['Experience'] - user_experience)
    df['exp_score'] = df['exp_diff'].apply(lambda x: max(0, 1 - x / 10))

    df['score'] = (
        0.4 * df['skill_match'] +
        0.2 * df['location_match'] +
        0.1 * df['exp_score'] +
        0.3 * df['similarity']
    )

    top_recommendations = df.sort_values(by='score', ascending=False).head(10)

    st.subheader("✅ Top Matches")
    for _, row in top_recommendations.iterrows():
        st.markdown(f"**{row['Title']}** at *{row['Company']}*")
        st.markdown(f"📍 {row['Location']} | 💼 {row['Experience']} years")
        st.markdown(f"🛠️ Skills: {', '.join(row['Skills_List'])}")
        st.markdown(f"🔗 [Apply Here]({row['Link']})")
        st.markdown(f"🌟 Score: {row['score']:.2f} | 🧠 Skill Match: {row['skill_match']:.2f}")
        st.markdown("---")

    st.download_button(
        label="📅 Download CSV",
        data=top_recommendations.to_csv(index=False),
        file_name='recommendations.csv',
        mime='text/csv'
    )
