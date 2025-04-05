# 🧠 Internship Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-red?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow?logo=scikit-learn)](https://scikit-learn.org/)
[![Last Updated](https://img.shields.io/badge/Updated-April%205%2C%202025-success)](https://naukri.com)
[![License](https://img.shields.io/badge/License-MIT-informational)](LICENSE)

A personalized internship recommender system built using **Streamlit**, **scikit-learn**, and data scraped from **Naukri.com** as of **April 5, 2025**. This app helps users find the most relevant internships based on their **skills**, **experience**, **preferred job role**, and **location**.

---

## 📌 Features

- 🔍 Scrapes and processes live internship listings from Naukri.com  
- 🧽 Data cleaning, skill parsing, and experience normalization  
- 🛠️ Multi-feature content-based filtering using:
  - TF-IDF for job titles
  - MultiLabelBinarizer for skills
  - One-hot encoding for locations
  - Scaled numeric input for experience  
- 🧮 Cosine similarity-based recommendation engine with a custom scoring system  
- 💡 Interactive Streamlit UI for real-time input and suggestions  
- ✅ Evaluation support for accuracy against test user profiles  

---

## 🗂 Dataset

- **Source:** [Naukri.com](https://www.naukri.com)
- **Scraped On:** April 5, 2025  
- **Fields Collected:**  
  - `Title`, `Company`, `Location`, `Experience`, `Skills`, `Link`  
- **Processed File:** `processed_data.csv`  

---

## 🚀 How It Works

### 1. **Feature Engineering**
- **Title:** Transformed using TF-IDF vectorization  
- **Skills:** Parsed into lists and binarized using `MultiLabelBinarizer`  
- **Location:** One-hot encoded  
- **Experience:** Normalized numeric scale via `StandardScaler`  

### 2. **User Input Collection**
- Preferred job title  
- Skillset (comma-separated)  
- Preferred location  
- Years of experience  

### 3. **Recommendation Pipeline**
- Converts user input into a vector using the same transformations  
- Computes **cosine similarity** between user vector and internship postings  
- Calculates a custom **score** based on:
  - `Skill Match` (50%)
  - `Location Match` (20%)
  - `Experience Similarity` (10%)
  - `Content Similarity` (20%)

### 4. **Output**
- Displays the **Top 10 Matches** ranked by score  
- Each card includes: Title, Company, Location, Required Skills, Experience, and Apply Link  

---

## 💻 How to Run

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the app
streamlit run app.py
```

##  **Evaluation (Optional)**
An additional script evaluates the recommender system’s accuracy using a test dataset of user profiles. It checks if the top recommended internships match their desired roles.

Accuracy on test set: ~XX.XX% (update after running evaluation script)

##  🤝 Contributing
Suggestions, issues, or contributions are welcome. Open a pull request or raise an issue!

## 📬 Contact
For any queries or collaboration:
- Ekaansh Sawaria
- Manipal University Jaipur




