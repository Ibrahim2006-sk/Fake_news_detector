import streamlit as st
import pandas as pd
import altair as alt
import requests
from bs4 import BeautifulSoup
import fitz
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==========================================
# STREAMLIT CONFIG
# ==========================================
st.set_page_config(page_title="Fake News Detector – Offline Ultra Pro", layout="wide")

st.title("📰 Fake News Detector – Offline Ultra Pro Version")
st.write("This version works **100% offline**. No HuggingFace model, no internet downloads.")

# ==========================================
# SAMPLE TRAINING DATA (LOCAL MODEL)
# ==========================================
fake_samples = [
    "Government confirms aliens landed on earth yesterday.",
    "Scientists discover immortality drug.",
    "Breaking news: celebrity cloned secretly.",
    "NASA reveals world ending next month in secret document.",
    "Actor replaced by AI clone in upcoming movie.",
]

real_samples = [
    "Prime Minister announces new economic reforms.",
    "RBI releases latest monetary policy update.",
    "India launches new satellite successfully.",
    "WHO reports improvement in global vaccination coverage.",
    "Government releases education sector report for 2024.",
]

X_train = fake_samples + real_samples
y_train = [0] * len(fake_samples) + [1] * len(real_samples)  # 0=fake, 1=real

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# ==========================================
# HELPERS
# ==========================================
def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def predict(text):
    vec = vectorizer.transform([clean(text)])
    proba = model.predict_proba(vec)[0]
    return float(proba[0]), float(proba[1])  # fake, real


def split_sentences(text):
    return re.split(r"(?<=[.!?]) +", text)


def extract_text_from_url(url):
    try:
        page = requests.get(url, timeout=10)
        soup = BeautifulSoup(page.text, "html.parser")
        paras = soup.find_all("p")
        return "\n".join([p.text for p in paras])
    except:
        return None


def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except:
        return None


def heat_color(prob):
    r = int(prob * 255)
    g = int((1 - prob) * 200)
    return f"rgb({r},{g},80)"


# ==========================================
# INPUT TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📝 Paste Text", "🔗 URL", "📄 PDF"])

text_input = ""

with tab1:
    text_input = st.text_area("Paste Article Text:", height=300)

with tab2:
    url = st.text_input("Enter URL:")
    if url:
        extracted = extract_text_from_url(url)
        if extracted:
            st.success("Article extracted!")
            text_input = extracted
        else:
            st.error("Could not extract article.")

with tab3:
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file:
        extracted = extract_text_from_pdf(pdf_file)
        if extracted:
            st.success("PDF extracted!")
            text_input = extracted
        else:
            st.error("PDF extraction failed.")


# ==========================================
# ANALYZE BUTTON
# ==========================================
if st.button("🚀 Run Offline Fake News Analysis"):
    if not text_input.strip():
        st.warning("Please enter some text.")
        st.stop()

    text = text_input[:6000]

    fake_score, real_score = predict(text)
    credibility = real_score * 100

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Overall Verdict")
        if fake_score > real_score:
            st.error(f"⚠️ Fake News Likely ({fake_score*100:.2f}% fake)")
        else:
            st.success(f"✅ Real News Likely ({credibility:.2f}% credible)")

    with col2:
        st.subheader("Credibility Gauge")
        color = "green" if credibility > 70 else "orange" if credibility > 40 else "red"
        st.markdown(
            f"""
            <div style='padding:20px;background:{color};color:white;
                text-align:center;font-size:26px;border-radius:10px;'>
                {credibility:.2f}% Credible
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # SENTENCE ANALYSIS
    st.subheader("Sentence-Level Fake Probability")
    sentences = split_sentences(text)
    results = []

    for s in sentences:
        if s.strip():
            fs, rs = predict(s)
            results.append({"sentence": s, "fake": fs, "real": rs})

    df = pd.DataFrame(results)

    # Heatmap
    for _, row in df.iterrows():
        st.markdown(
            f"""
            <div style='padding:10px;background:{heat_color(row["fake"])};
            border-radius:5px;color:white;margin-bottom:6px;'>
                {row["sentence"]}<br>
                <small>Fake Probability: {row["fake"]:.2f}</small>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("Fake Probability Chart")
    chart = alt.Chart(df.reset_index()).mark_bar().encode(
        x="index",
        y="fake",
        tooltip=["sentence", "fake"]
    ).properties(width=800, height=300)

    st.altair_chart(chart, use_container_width=True)

    # CSV DOWNLOAD
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Analysis CSV",
                       data=csv, file_name="offline_fake_news_analysis.csv")

    st.subheader("Full Article Text")
    st.write(text_input)
