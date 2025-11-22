import streamlit as st
from transformers import pipeline
import textwrap
import time
import pandas as pd
import altair as alt
import re
import random 

# --- CONFIGURATION CONSTANTS ---
MIN_CHARACTERS = 2
MAX_CHARACTERS = 500

# --- SAMPLE TEXTS FOR TESTING ---
SAMPLE_TEXTS = [
    "I'm so excited about the new project! It looks brilliant and I think we're going to achieve wonderful things together. This is a genuinely positive and supportive message.",
    "You are a complete idiot, a worthless failure, and you should just shut up and quit this platform right now. You are an absolute disgrace.",
    "Your plan is bad. Your execution is poor. I hope you get doxxed and exposed for who you really are. I will find you.",
    "This is such a stupid, pathetic, and frankly awful post. Go back to school, you ignorant fool, you suck.",
    "The weather today is beautiful, and I really appreciate the hard work everyone put into the presentation. Good job! Keep up the excellent work.",
    "I will find you and attack you. You will regret every word you ever wrote here. This is a severe threat to your well-being.",
    "Get your disgusting, trashy comments out of this chat, you absolute moron. You are a joke. You're completely pathetic.",
    "The result was crap, but you are still a brilliant programmer and I know you can fix this problem easily.",
]

# --- Model Loading & Caching ---
@st.cache_resource(show_spinner=False)
def load_model():
    """Load and cache HuggingFace toxic-bert model pipeline."""
    try:
        clf = pipeline("text-classification", model="unitary/toxic-bert", return_all_scores=True, trust_remote_code=False)
        return clf
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

classifier = load_model()

# --- HYBRID LEXICON DEFINITION ---
LEXICON_RISK = {
    "SEVERE_TOXIC": {
        "words": ["kill", "doxx", "doxxed", "dox", "doxed", "cyberstalking", "stalking", "stalker", 
                  "threaten", "threat", "suicide", "k-y-s", "kys", "kill yourself", "die",
                  "rape", "r-a-p-e", "murder", "assault", "attack", "harm you", "hurt you",
                  "find you", "come after", "destroy you", "end you"],
        "weight": 0.35
    },
    "IDENTITY_HATE": {
        "words": ["do not belong here", "not welcome", "your kind", "that kind of background",
                  "those people", "go back", "get out", "shouldn't be allowed", "stay away",
                  "displace", "foreign", "illegal"],
        "weight": 0.25
    },
    "INSULT": {
        "words": ["idiot", "stupid", "dumb", "ugly", "useless", "worthless", "loser", "failure", 
                  "pathetic", "disgusting", "trash", "garbage", "scum", "moron", "fool",
                  "incompetent", "terrible", "awful", "horrible", "poor", "bad"],
        "weight": 0.15
    },
    "OBSCENE": {
        "words": ["fuck", "shit", "bitch", "ass", "damn", "hell", "bastard", "crap",
                  "piss", "dick", "cock", "pussy", "whore", "slut"],
        "weight": 0.20
    },
    "POSITIVE": {
        "words": ["amazing", "brilliant", "wonderful", "excellent", "respect", "support", "kind", 
                  "fantastic", "friendly", "awesome", "help", "great", "love", "beautiful",
                  "appreciate", "thank", "good", "nice", "perfect"],
        "weight": -0.20
    }
}

# --- CORE LOGIC FUNCTIONS ---
def map_label_to_category(label: str) -> str:
    """Map raw model labels to user-friendly categories for visualization."""
    label = label.lower().replace("label_", "")
    mapping = {
        "toxic": "Toxic / Abusive",
        "severe_toxic": "Severe Harassment",
        "obscene": "Obscene / Vulgar",
        "threat": "Threatening",
        "insult": "Insult / Bullying",
        "identity_hate": "Hate Speech"
    }
    return mapping.get(label, label.title())

def get_safety_message(scores: list, summary: dict) -> tuple[str, str, str]:
    """Assesses overall safety based on hybrid scores and lexicon data."""
    if not scores:
        return "N/A", "Cannot analyze empty results.", "info"
    
    # Check for severe lexicon hits first
    if summary.get("severe_lexicon_hit"):
        triggered_cat = summary.get('triggered_category', 'unknown').replace('_', ' ').title()
        risk_msg = "🚨 **EXTREME RISK (Lexicon Override)**"
        msg = f"{risk_msg} | Critical threat word detected: **{triggered_cat}**"
        return "🛑 **THREAT DETECTED**", msg, "error"
    
    top = scores[0]
    top_label = top["label"].lower().replace("label_", "")
    top_score = top["score"]
    
    if top_score < 0.40:
        return "✅ **Content is Safe**", "This message is classified as non-toxic.", "success"
    
    if top_label in ["severe_toxic", "threat", "identity_hate"]:
        if top_score > 0.70:
            risk_msg = "🚨 **EXTREME RISK**"
        else:
            risk_msg = "🚨 **HIGH RISK**"
        msg = f"{risk_msg} | Detected **{map_label_to_category(top_label)}** with {round(top_score*100)}% confidence."
        return "🛑 **THREAT DETECTED**", msg, "error"
    
    if top_score >= 0.40:
        if top_score >= 0.60:
            risk_msg = "🚨 **HIGH RISK**"
            status = "error"
        else:
            risk_msg = "⚠️ **MODERATE RISK**"
            status = "warning"
        msg = f"{risk_msg} | Detected **{map_label_to_category(top_label)}** with {round(top_score*100)}% confidence."
        return "⚠️ **CAUTION REQUIRED**", msg, status
    
    return "✅ **Content is Safe**", "This message seems safe.", "success"

def get_lexicon_score_adjustment(text: str, scores: list) -> tuple[list, dict]:
    """Applies lexicon-based adjustments to the BERT scores."""
    norm_text = text.lower()
    summary = {
        "total_adjustment": 0.0,
        "triggered_category": "unknown",
        "severe_lexicon_hit": False,
        "detected_words": [],
        "risk_word_count": 0,
        "positive_word_count": 0
    }
    
    adjustment = 0.0
    triggered_category = "unknown"
    max_weight = 0.0
    
    # Calculate adjustments and track detected words
    for key, data in LEXICON_RISK.items():
        for word in data["words"]:
            pattern = r'\b' + re.escape(word) + r'\b' if len(word.split()) == 1 else re.escape(word)
            if re.search(pattern, norm_text):
                adjustment += data["weight"]
                summary["detected_words"].append(word)
                
                if data["weight"] > 0:  # Risk word
                    summary["risk_word_count"] += 1
                    if data["weight"] > max_weight:
                        max_weight = data["weight"]
                        triggered_category = key.lower()
                        if key == "SEVERE_TOXIC":
                            summary["severe_lexicon_hit"] = True
                else:  # Positive word
                    summary["positive_word_count"] += 1
    
    summary["total_adjustment"] = adjustment
    summary["triggered_category"] = triggered_category
    
    # Apply adjustments to scores
    adjusted_scores = []
    for item in scores:
        label = item["label"].lower().replace("label_", "")
        score = item["score"]
        new_score = score
        
        if triggered_category != "unknown" and adjustment > 0:
            if label == triggered_category:
                new_score = min(1.0, score + max_weight)
            elif label == "toxic":
                new_score = min(1.0, score + (max_weight * 0.5))
        
        if adjustment < 0 and label not in ["severe_toxic", "threat"]:
            new_score = max(0.0, score + adjustment)
        
        adjusted_scores.append({"label": item["label"], "score": new_score})
    
    adjusted_scores = sorted(adjusted_scores, key=lambda x: x["score"], reverse=True)
    return adjusted_scores, summary

def classify_text(text: str):
    """Orchestrates model execution, lexicon scoring, and handles input validation."""
    if not text or not text.strip():
        return None, None
    
    text = text.strip()
    
    if len(text) < MIN_CHARACTERS:
        st.warning(f"⚠️ Message too short for reliable analysis (minimum {MIN_CHARACTERS} characters).")
        return None, None
    
    if len(text) > MAX_CHARACTERS:
        st.info(f"ℹ️ Long message detected. Analyzing first {MAX_CHARACTERS} characters.")
        text = text[:MAX_CHARACTERS]
    
    with st.spinner("🔍 Analyzing text with **Hybrid AI**..."):
        time.sleep(0.3)
        try:
            bert_scores = classifier(text)[0]
            adjusted_scores, summary = get_lexicon_score_adjustment(text, bert_scores)
            return adjusted_scores, summary
        except Exception as e:
            st.error(f"❌ Model error: {str(e)}")
            return None, None

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="Project SHIELD – Hybrid AI Harassment Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme with animated background
st.markdown("""
<style>
    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: linear-gradient(-45deg, #1f2833, #2c3a47, #151a21, #0c1015);
        background-size: 400% 400%;
        animation: gradient-animation 25s ease infinite;
        color: white;
    }
    
    .main { background-color: transparent !important; }
    section[data-testid="stSidebar"] { background: #1f2833 !important; }

    .stylish-title {
        font-family: 'Montserrat', 'Open Sans', sans-serif; 
        font-weight: 900;
        font-size: 3.8rem; 
        text-align: center;
        background: -webkit-linear-gradient(45deg, #00BFFF, #1E90FF); 
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 8px rgba(0, 191, 255, 0.4); 
    }
    .main-header { text-align: center; padding: 1rem 0; }
    
    textarea { background-color: #333333 !important; color: white !important; }
    h1, h2, h3, h4, h5, h6 { color: #f0f0f0; }
</style>
""", unsafe_allow_html=True)

# Custom Altair Theme for dark mode
def dark_theme():
    return {
        "config": {
            "background": "transparent",
            "title": {"color": "white", "fontSize": 14},
            "style": {"guide-label": {"fill": "white"}, "guide-title": {"fill": "white"}},
            "axis": {"domainColor": "#666666", "gridColor": "#444444", "tickColor": "#666666"},
        }
    }

alt.themes.register("dark_theme", dark_theme)
alt.themes.enable("dark_theme")

# --- UI Content ---
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.markdown('<h1 class="stylish-title">🛡️ Project SHIELD 3.0</h1>', unsafe_allow_html=True)
st.subheader("Safe Harassment & Intimidation Elimination using Learning-based Detection")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
<p style='color: #f0f0f0;'>Paste any message, comment, or chat text below. The analysis runs <b>automatically</b> as you type.  
<i>We are here to help.</i> (Min: **{MIN_CHARACTERS}** chars | Max: **{MAX_CHARACTERS}** chars)</p>
""", unsafe_allow_html=True)

# Session State
if 'user_text_area' not in st.session_state:
    st.session_state.user_text_area = ""

def set_sample_text():
    sample_text = random.choice(SAMPLE_TEXTS)
    st.session_state.user_text_area = sample_text

def clear_text():
    st.session_state.user_text_area = ""

# Input
user_text = st.text_area(
    f"🔹 **Message to Analyze** (Min {MIN_CHARACTERS} / Max {MAX_CHARACTERS} characters):",
    value=st.session_state.user_text_area,
    height=150,
    placeholder=f"Example: \"You're terrible\" (Minimum {MIN_CHARACTERS} characters required)",
    key="user_text_area",
    help=f"Enter text between {MIN_CHARACTERS} and {MAX_CHARACTERS} characters"
)

col1, col2 = st.columns([1, 3])
with col1:
    st.button("📌 **Try Sample Text**", use_container_width=True, on_click=set_sample_text)
with col2:
    st.button("🗑️ **Clear Text**", use_container_width=True, on_click=clear_text)

# Analysis
scores, summary = classify_text(user_text)

if scores and summary:
    status_title, safety_message, status_color = get_safety_message(scores, summary)
    st.markdown("---")
    st.subheader(f"✨ **Analysis Result: {status_title}**")
    
    if status_color == "error":
        st.error(safety_message)
    elif status_color == "warning":
        st.warning(safety_message)
    else:
        st.success(safety_message)
        st.balloons()

    st.markdown("### 📊 Detailed Toxicity Breakdown (Hybrid Score)")
    st.caption("Scores are **augmented** by the Lexicon Engine for enhanced detection.")

    # Visualization
    data = []
    for item in scores:
        if item["score"] > 0.05:
            data.append({
                "Category": map_label_to_category(item["label"]),
                "Score": item["score"],
                "Confidence": round(item["score"] * 100, 1)
            })

    df = pd.DataFrame(data)

    if df.empty:
        st.info("📊 **No Toxicity Detected:** All risk scores below 5% threshold.")
    else:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("Confidence", title="Augmented Confidence (%)", axis=alt.Axis(format=".1f")),
            y=alt.Y("Category", sort=alt.EncodingSortField(field="Confidence", op="max", order="descending"), title="Toxicity Category"),
            tooltip=["Category", "Confidence"],
            color=alt.Color("Score", 
                scale=alt.Scale(
                    domain=[0.0, 0.3, 0.6, 1.0],
                    range=['#42a5f5', '#ffa726', '#ef5350', '#c62828']
                ),
                legend=None
            )
        ).properties(
            title="Hybrid Toxicity Scores by Category"
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
    
    # Lexicon Summary - FIXED VERSION
    if summary.get("detected_words"):
        st.markdown("#### 🔬 **Lexicon Engine Report**")
        
        col_neg, col_pos = st.columns(2)
        with col_neg:
            triggered_cat = summary.get('triggered_category', 'unknown')
            trigger_display = triggered_cat.replace('_', ' ').title() if triggered_cat != 'unknown' else "N/A"
            st.metric(
                label="Risk Words Detected", 
                value=summary.get("risk_word_count", 0),
                delta=f"Category: {trigger_display}"
            )
        with col_pos:
            st.metric(
                label="Positive Words (FP Suppression)", 
                value=summary.get("positive_word_count", 0),
                delta=f"Adjustment: {round(summary.get('total_adjustment', 0), 3)}"
            )
        
        with st.expander("🔍 View Detected Keywords"):
            st.write("**Flagged Words:**", ", ".join(summary["detected_words"][:10]))

    st.markdown("---")
    st.markdown("### 🧭 **Immediate Recommended Actions**")
    
    if status_color in ["error", "warning"]:
        st.error("⚠️ **Protect yourself immediately:**")
        st.markdown("""
        <ul style='color: white;'>
            <li><b>📸 Document:</b> Screenshot and save this message as evidence</li>
            <li><b>🚫 Block:</b> Immediately block the sender's account</li>
            <li><b>📢 Report:</b> Use platform reporting tools to flag this content</li>
            <li><b>💬 Seek Support:</b> Talk to a trusted adult, teacher, or counselor</li>
            <li><b>🔒 Privacy:</b> Review and strengthen your privacy settings</li>
        </ul>
        """, unsafe_allow_html=True)
    else:
        st.info("✅ Everything looks safe. Remember, we're here if you need us.")

else:
    st.info(f"💡 **Welcome to Project SHIELD.** Enter a message above (minimum **{MIN_CHARACTERS}** characters) to begin analysis.")

# Sidebar
st.sidebar.title("ℹ️ **About Project SHIELD**")
st.sidebar.markdown("""
<p style='color: #f0f0f0;'><b>Project SHIELD 3.0</b> (Safe Harassment & Intimidation Elimination using Learning-based Detection) is your <b>Digital Guardian</b>.</p>

<p style='color: #f0f0f0;'><b>Hybrid AI System:</b></p>
<ul style='color: #f0f0f0;'>
    <li><b>Deep Learning (BERT):</b> Advanced language understanding</li>
    <li><b>Rule-Based Lexicon:</b> Enhanced keyword detection</li>
    <li><b>Dual-Layer Protection:</b> Maximum accuracy</li>
</ul>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📞 **Crisis Support Resources**")
st.sidebar.markdown("""
<ul style='color: #f0f0f0;'>
    <li><b>National Suicide Prevention:</b> 988</li>
    <li><b>Crisis Text Line:</b> Text HOME to 741741</li>
    <li><b>Cyberbullying Research:</b> cyberbullying.org</li>
    <li><b>RAINN (Sexual Assault):</b> 1-800-656-4673</li>
</ul>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("🚀 Built by **Team Circuit 404**")
st.sidebar.caption("Hackathon 2025 • EEE Branch")

# Footer
st.markdown("---")
st.caption("⚡ Powered by Hybrid AI | Your Digital Guardian | © 2025 Team Circuit 404")