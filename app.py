# app.py
import streamlit as st
import pandas as pd
import altair as alt
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import fitz  # pymupdf
import numpy as np

# ---------------------------
# Streamlit config
# ---------------------------
st.set_page_config(page_title="Fake News Detector — Offline + FactCheck", layout="wide")
st.title("📰 Fake News Detector — Offline + Local Fact-Checking")

# ---------------------------
# Local Fact-Check Database (editable)
# Add known debunked/verified claims here. Expand as needed.
# Each entry: 'claim' (short text), 'verdict' ('TRUE'/'FALSE'/'MIXED'), 'source', 'note'
# ---------------------------
LOCAL_FACTCHECK_DB = [
    {
        "claim": "Government confirms aliens have landed on earth.",
        "verdict": "FALSE",
        "source": "ExampleFactCheck",
        "note": "No credible evidence; debunked by major outlets."
    },
    {
        "claim": "Scientists discover immortality drug.",
        "verdict": "FALSE",
        "source": "ExampleFactCheck",
        "note": "Sensational claim with no peer-reviewed support."
    },
    {
        "claim": "India launches new satellite successfully.",
        "verdict": "TRUE",
        "source": "TrustedNewsArchive",
        "note": "Supported by official space agency release."
    },
    {
        "claim": "RBI releases latest monetary policy update.",
        "verdict": "TRUE",
        "source": "TrustedNewsArchive",
        "note": "Official press release exists."
    },
]

# ---------------------------
# Small set of trusted articles (editable)
# Use representative real-news sample texts here for "support" matching
# ---------------------------
TRUSTED_ARTICLES = [
    {
        "title": "RBI releases monetary policy statement 2025",
        "text": "The Reserve Bank of India today announced its monetary policy updates including repo rate and liquidity measures."
    },
    {
        "title": "India launches latest satellite",
        "text": "India successfully launched its newest earth observation satellite into orbit, as confirmed by the space agency."
    },
    {
        "title": "WHO reports on vaccine coverage",
        "text": "The World Health Organization reported improved global vaccination coverage in its latest report."
    },
]

# ---------------------------
# Utility helpers
# ---------------------------
def clean_text(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"http\S+", "", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return s.strip()

def split_sentences(text: str):
    # naive sentence splitter
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sents:
        return [text.strip()]
    return sents

# ---------------------------
# Build vectorizers for local DB
# Vectorizer trained on all fact-claims + trusted articles texts
# ---------------------------
all_claim_texts = [clean_text(e["claim"]) for e in LOCAL_FACTCHECK_DB]
all_trusted_texts = [clean_text(a["text"]) for a in TRUSTED_ARTICLES]

# Combined corpus for TF-IDF training (so similarity is consistent)
CORPUS = all_claim_texts + all_trusted_texts
if len(CORPUS) == 0:
    CORPUS = ["placeholder"]

vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english').fit(CORPUS)

# Precompute vectors for claims and trusted articles
claim_vectors = vectorizer.transform(all_claim_texts) if all_claim_texts else None
trusted_vectors = vectorizer.transform(all_trusted_texts) if all_trusted_texts else None

# ---------------------------
# Predict / Fact-check logic
# ---------------------------
def local_factcheck_sentence(sentence: str,
                             claim_vecs,
                             trusted_vecs,
                             claim_threshold=0.65,
                             trusted_threshold=0.45):
    """
    For a given sentence, compute:
      - best matching local fact-check claim (if similarity > claim_threshold)
      - best matching trusted article (if similarity > trusted_threshold)
    Returns a dict with verdict, matched_claim (if any), matched_trusted (if any), scores.
    """
    s_clean = clean_text(sentence)
    s_vec = vectorizer.transform([s_clean])

    result = {
        "sentence": sentence,
        "cleaned": s_clean,
        "matched_claim": None,
        "matched_claim_score": 0.0,
        "matched_trusted": None,
        "matched_trusted_score": 0.0,
        "final_verdict": "UNVERIFIED"  # default
    }

    # Match against fact-check claims (debunks/verifications)
    if claim_vecs is not None and claim_vecs.shape[0] > 0:
        sims = cosine_similarity(s_vec, claim_vecs)[0]  # shape (n_claims,)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        result["matched_claim_score"] = best_score
        if best_score >= claim_threshold:
            # strong match to a known fact-check claim
            matched = LOCAL_FACTCHECK_DB[best_idx]
            result["matched_claim"] = matched
            # final verdict from DB: TRUE => REAL, FALSE => FAKE, MIXED => UNVERIFIED/MIXED
            if matched["verdict"].upper() in ("FALSE", "FALSEY", "FAKE"):
                result["final_verdict"] = "FAKE"
            elif matched["verdict"].upper() in ("TRUE","TRUEY"):
                result["final_verdict"] = "REAL"
            else:
                result["final_verdict"] = "MIXED"
            # If claim matched, we can return (still will also compute trusted)
    # Match against trusted articles for supporting evidence
    if trusted_vecs is not None and trusted_vecs.shape[0] > 0:
        tsims = cosine_similarity(s_vec, trusted_vecs)[0]
        tb_idx = int(np.argmax(tsims))
        tb_score = float(tsims[tb_idx])
        result["matched_trusted_score"] = tb_score
        if tb_score >= trusted_threshold:
            matched_t = TRUSTED_ARTICLES[tb_idx]
            result["matched_trusted"] = matched_t
            # If already matched_claim with FALSE, keep FAKE (debunk override)
            if result["final_verdict"] == "UNVERIFIED":
                # supported by trusted source => REAL
                result["final_verdict"] = "REAL"

    return result

# ---------------------------
# UI: Input tabs (text / url / pdf)
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📝 Paste Text", "🔗 Analyze URL", "📄 Upload PDF"])
text_input = ""

with tab1:
    text_input = st.text_area("Paste article text here:", height=300)

with tab2:
    url = st.text_input("Enter article URL:")
    if url:
        try:
            with st.spinner("Fetching URL..."):
                r = requests.get(url, timeout=8)
                soup = BeautifulSoup(r.text, "html.parser")
                paragraphs = soup.find_all("p")
                text_input = "\n".join(p.get_text() for p in paragraphs)
                if not text_input.strip():
                    st.error("Couldn't extract article text from this URL.")
                else:
                    st.success("URL extracted into text area.")
        except Exception as e:
            st.error("Failed to fetch URL: " + str(e))

with tab3:
    pdf_file = st.file_uploader("Upload PDF (optional)", type=["pdf"])
    if pdf_file:
        try:
            with st.spinner("Extracting PDF text..."):
                doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
                txt = ""
                for p in doc:
                    txt += p.get_text()
                if txt.strip():
                    text_input = txt
                    st.success("PDF text extracted into text area.")
                else:
                    st.error("No text found in PDF.")
        except Exception as e:
            st.error("PDF extraction failed: " + str(e))

# ---------------------------
# Settings in sidebar
# ---------------------------
st.sidebar.header("Fact-check settings")
claim_thresh = st.sidebar.slider("Claim match threshold (local fact DB)", 0.4, 0.9, 0.65, 0.01)
trusted_thresh = st.sidebar.slider("Trusted-article match threshold", 0.2, 0.8, 0.45, 0.01)
max_sentences = st.sidebar.number_input("Max sentences to analyze (for very long text)", min_value=5, max_value=400, value=120, step=5)

st.sidebar.markdown("---")
st.sidebar.write("Edit `LOCAL_FACTCHECK_DB` and `TRUSTED_ARTICLES` in app.py to expand local coverage.")

# ---------------------------
# Analyze button
# ---------------------------
if st.button("🔎 Analyze & Local Fact-Check"):
    if not text_input or not text_input.strip():
        st.warning("Please paste or provide an article (text, URL or PDF).")
        st.stop()

    article = text_input.strip()
    sents = split_sentences(article)
    sents = sents[:max_sentences]

    # Run local factcheck for each sentence
    results = []
    for sent in sents:
        if len(sent) < 8:
            continue
        res = local_factcheck_sentence(sent, claim_vectors, trusted_vectors,
                                       claim_threshold=claim_thresh,
                                       trusted_threshold=trusted_thresh)
        results.append(res)

    # Aggregate article-level verdict:
    # If any sentence is FAKE (debunked), article is likely FAKE.
    # Else if many sentences REAL or supported, article likely REAL.
    n_fake = sum(1 for r in results if r["final_verdict"] == "FAKE")
    n_real = sum(1 for r in results if r["final_verdict"] == "REAL")
    n_unv = sum(1 for r in results if r["final_verdict"] == "UNVERIFIED")
    total = len(results) or 1

    # Simple heuristic scoring
    fake_ratio = n_fake / total
    real_ratio = n_real / total

    if fake_ratio >= 0.15:
        article_verdict = "FAKE"
    elif real_ratio >= 0.5:
        article_verdict = "REAL"
    else:
        article_verdict = "UNVERIFIED"

    # Present summary
    st.markdown("## Summary")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.metric("Sentences analyzed", len(results))
        st.metric("Debunked sentences", n_fake)
    with col2:
        st.metric("Supported sentences", n_real)
        st.metric("Unverified sentences", n_unv)
    with col3:
        if article_verdict == "REAL":
            st.success("🟢 ARTICLE VERDICT: REAL")
        elif article_verdict == "FAKE":
            st.error("🔴 ARTICLE VERDICT: FAKE")
        else:
            st.info("🟠 ARTICLE VERDICT: UNVERIFIED")

    st.markdown("---")

    # Show matched fact-checks (strong matches)
    matched_claims = [r for r in results if r["matched_claim"] is not None and r["matched_claim_score"] >= claim_thresh]
    if matched_claims:
        st.subheader("Matched Local Fact-Checks (strong matches)")
        for r in matched_claims:
            c = r["matched_claim"]
            st.markdown(f"**Claim match:** {r['sentence']}")
            if c:
                verdict = c.get("verdict","")
                if verdict.upper() == "FALSE":
                    st.error(f"Local DB Verdict: {verdict} — {c.get('note','')}")
                elif verdict.upper() == "TRUE":
                    st.success(f"Local DB Verdict: {verdict} — {c.get('note','')}")
                else:
                    st.info(f"Local DB Verdict: {verdict} — {c.get('note','')}")
                st.write(f"Source: {c.get('source')}")
                st.write(f"Similarity: {r['matched_claim_score']:.3f}")
            st.markdown("---")

    # Show supported sentences by trusted articles
    supported = [r for r in results if r["matched_trusted"] is not None and r["matched_trusted_score"] >= trusted_thresh]
    if supported:
        st.subheader("Supported by Trusted Articles (possible corroboration)")
        for r in supported:
            t = r["matched_trusted"]
            st.markdown(f"**Sentence:** {r['sentence']}")
            st.write(f"Matched trusted article: **{t['title']}** (score: {r['matched_trusted_score']:.3f})")
            st.write(t["text"][:300] + ("…" if len(t["text"]) > 300 else ""))
            st.markdown("---")

    # Sentence-level display with badges
    st.subheader("Sentence-level results")
    rows = []
    for r in results:
        badge = "🟠 UNVERIFIED"
        if r["final_verdict"] == "FAKE":
            badge = "🔴 FAKE"
        elif r["final_verdict"] == "REAL":
            badge = "🟢 REAL"

        rows.append({
            "sentence": r["sentence"],
            "verdict": badge,
            "claim_sim": r["matched_claim_score"],
            "trusted_sim": r["matched_trusted_score"]
        })

    df = pd.DataFrame(rows)
    # Show table
    st.dataframe(df[["verdict","claim_sim","trusted_sim","sentence"]], use_container_width=True)

    # Highlight sentences (colored boxes)
    st.markdown("---")
    st.subheader("Highlights")
    for r in results:
        color = "#f0ad4e"  # default orange for unverified
        if r["final_verdict"] == "REAL":
            color = "#6cc24a"  # green
        elif r["final_verdict"] == "FAKE":
            color = "#e85a5a"  # red

        sim_text = ""
        if r["matched_claim"]:
            sim_text += f"Matched claim ({r['matched_claim_score']:.2f}) from {r['matched_claim']['source']}. "
        if r["matched_trusted"]:
            sim_text += f"Supported by {r['matched_trusted']['title']} ({r['matched_trusted_score']:.2f})."

        st.markdown(
            f"<div style='background:{color};padding:12px;border-radius:10px;color:white;margin-bottom:8px;'>"
            f"<b>{r['final_verdict']}</b> — {r['sentence']}<br><small>{sim_text}</small>"
            f"</div>",
            unsafe_allow_html=True
        )

    # CSV download of sentence results
    out_df = pd.DataFrame([{
        "sentence": r["sentence"],
        "verdict": r["final_verdict"],
        "matched_claim_score": r["matched_claim_score"],
        "matched_trusted_score": r["matched_trusted_score"],
        "matched_claim": r["matched_claim"]["claim"] if r["matched_claim"] else "",
        "matched_claim_verdict": r["matched_claim"]["verdict"] if r["matched_claim"] else "",
        "matched_trusted_title": r["matched_trusted"]["title"] if r["matched_trusted"] else ""
    } for r in results])
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download sentence fact-check CSV", data=csv_bytes, file_name="sentence_factcheck.csv")

    st.success("Local fact-checking complete. Remember: extend LOCAL_FACTCHECK_DB & TRUSTED_ARTICLES to increase coverage.")
