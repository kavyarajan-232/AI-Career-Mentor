# ============================================================
# FILE: app/app.py
# PURPOSE: Streamlit frontend — Career Mentor & Growth Planner
# RUN:  streamlit run app/app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI Career Mentor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title   { font-size:2.2rem; font-weight:700; color:#3266ad; text-align:center; }
    .sub-title    { font-size:1rem; color:#666; text-align:center; margin-bottom:1.5rem; }
    .career-card  { background:#f0f4ff; border-left:5px solid #3266ad;
                    padding:1rem 1.2rem; border-radius:8px; margin:0.5rem 0; }
    .score-badge  { display:inline-block; padding:4px 14px; border-radius:20px;
                    font-weight:600; font-size:1rem; }
    .green-bg     { background:#d4edda; color:#155724; }
    .blue-bg      { background:#cce5ff; color:#004085; }
    .amber-bg     { background:#fff3cd; color:#856404; }
    .section-head { font-size:1.2rem; font-weight:600; color:#3266ad; border-bottom:2px solid #3266ad;
                    padding-bottom:4px; margin:1.2rem 0 0.8rem; }
    .tip-box      { background:#fffbea; border:1px solid #ffe082; border-radius:6px;
                    padding:0.6rem 1rem; margin:0.4rem 0; font-size:0.95rem; }
    .roadmap-item { display:flex; align-items:center; gap:10px; padding:6px 0;
                    border-bottom:1px solid #e8e8e8; }
    .step-num     { background:#3266ad; color:white; border-radius:50%;
                    width:26px; height:26px; display:flex; align-items:center;
                    justify-content:center; font-weight:600; font-size:0.85rem; flex-shrink:0; }
</style>
""", unsafe_allow_html=True)


# ── Model Path ─────────────────────────────────────────────────
# NOTE: change this if you move the project folder
MODEL_DIR = r"C:\Users\LENOVO\OneDrive\Desktop\Career Mentor\Project File"


# ══════════════════════════════════════════════════════════════
# LOAD MODELS (cached)
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    with open(os.path.join(MODEL_DIR, "best_model.pkl"),      "rb") as f: model         = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "random_forest.pkl"),   "rb") as f: rf            = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "decision_tree.pkl"),   "rb") as f: dt            = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoders.pkl"),  "rb") as f: label_enc     = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"),          "rb") as f: scaler        = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_names.pkl"),   "rb") as f: feature_names = pickle.load(f)
    return model, rf, dt, label_enc, scaler, feature_names


try:
    model, rf, dt, label_enc, scaler, feature_names = load_models()
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    st.warning(f"⚠️ Models not found — run 02_train_model.py first. ({e})")


# ══════════════════════════════════════════════════════════════
# STATIC DATA (roadmaps, salaries, skills)
# ══════════════════════════════════════════════════════════════
ROADMAPS = {
    "Machine Learning Engineer": [
        ("Month 1", "Python Fundamentals", "Variables, loops, OOP, file handling"),
        ("Month 2", "NumPy & Pandas",       "Data manipulation, Series, DataFrames"),
        ("Month 2–3", "Statistics & Math",  "Probability, linear algebra, distributions"),
        ("Month 3–4", "Machine Learning",   "Sklearn: regression, classification, clustering"),
        ("Month 5",  "Deep Learning",       "TensorFlow / PyTorch, CNNs, RNNs"),
        ("Month 6",  "MLOps & Deployment",  "Flask API, Docker, model serving"),
        ("Month 6+", "Certifications",      "Google ML, Coursera Deep Learning Spec"),
    ],
    "Data Scientist": [
        ("Month 1",  "Python / R basics",       "Core programming for data tasks"),
        ("Month 1–2","SQL & Databases",          "Querying, joins, aggregations"),
        ("Month 2",  "Data Visualization",       "Matplotlib, Seaborn, Tableau"),
        ("Month 2–3","Statistics",               "Hypothesis testing, regression"),
        ("Month 4",  "Machine Learning",         "Regression, classification, clustering"),
        ("Month 5",  "Big Data Tools",           "Spark, Hadoop, cloud platforms"),
        ("Month 6",  "Portfolio & Certs",        "IBM Data Science cert, Kaggle projects"),
    ],
    "Cybersecurity Analyst": [
        ("Month 1",  "Networking Basics",   "TCP/IP, DNS, OSI model, HTTP"),
        ("Month 1–2","Linux Fundamentals",  "CLI, permissions, shell scripting"),
        ("Month 2–3","Security Concepts",   "CIA triad, threats, vulnerabilities"),
        ("Month 3–4","Ethical Hacking",     "Kali Linux, Metasploit, pen testing"),
        ("Month 5",  "SIEM & SOC Tools",    "Splunk, Wireshark, log analysis"),
        ("Month 6",  "Certifications",      "CompTIA Security+, CEH, OSCP path"),
    ],
    "Web Developer": [
        ("Month 1",  "HTML & CSS",      "Semantic HTML, Flexbox, Grid, responsive"),
        ("Month 1–2","JavaScript",       "DOM, ES6+, async/await, fetch API"),
        ("Month 3",  "React / Vue",      "Components, hooks, routing, state"),
        ("Month 3–4","Node.js & Express","REST APIs, middleware, auth"),
        ("Month 4–5","Databases",        "MongoDB, PostgreSQL, ORMs"),
        ("Month 5–6","Deployment",       "Docker, Vercel, AWS / GCP basics"),
    ],
    "Cloud Engineer": [
        ("Month 1",  "Linux & CLI",      "Shell scripting, sysadmin basics"),
        ("Month 1–2","Networking",        "VPC, subnets, load balancers, DNS"),
        ("Month 2–4","AWS / Azure / GCP","Core services: EC2, S3, Lambda, IAM"),
        ("Month 4–5","Docker & K8s",     "Containerization, orchestration"),
        ("Month 5",  "DevOps & CI/CD",   "GitHub Actions, Terraform, monitoring"),
        ("Month 6",  "Certifications",   "AWS Cloud Practitioner → Solutions Architect"),
    ],
    "Software Developer": [
        ("Month 1",  "Core Language",       "Python / Java / C++ — your primary language"),
        ("Month 1–2","Data Structures",     "Arrays, linked lists, trees, graphs"),
        ("Month 2–3","Algorithms",          "Sorting, searching, dynamic programming"),
        ("Month 3–4","System Design",       "Low-level & high-level design"),
        ("Month 4–5","Frameworks",          "Spring Boot / Django / FastAPI"),
        ("Month 5–6","Testing & Deployment","Unit testing, CI/CD, Docker"),
    ],
}


SALARY = {
    "Machine Learning Engineer": ("₹6–14 LPA", "₹18–35 LPA", "Very High 🚀"),
    "Data Scientist":            ("₹5–12 LPA", "₹15–30 LPA", "High 📈"),
    "Cybersecurity Analyst":     ("₹5–11 LPA", "₹15–28 LPA", "High 📈"),
    "Web Developer":             ("₹4–10 LPA", "₹12–25 LPA", "Moderate 📊"),
    "Cloud Engineer":            ("₹6–13 LPA", "₹18–32 LPA", "Very High 🚀"),
    "Software Developer":        ("₹5–12 LPA", "₹15–30 LPA", "High 📈"),
}


SKILLS_NEEDED = {
    "Machine Learning Engineer": ["Python","Scikit-learn","TensorFlow","Statistics","MLOps"],
    "Data Scientist":            ["Python","SQL","Pandas","Statistics","Tableau"],
    "Cybersecurity Analyst":     ["Networking","Linux","Security Tools","Cryptography","SIEM"],
    "Web Developer":             ["HTML/CSS","JavaScript","React","Node.js","MongoDB"],
    "Cloud Engineer":            ["Linux","AWS/Azure","Docker","Kubernetes","Terraform"],
    "Software Developer":        ["DSA","OOP","System Design","Git","Testing"],
}


INTERNSHIPS = {
    "Machine Learning Engineer": ["AI Research Intern","ML Engineer Intern","Data Science Intern"],
    "Data Scientist":            ["Data Analyst Intern","BI Intern","Data Science Intern"],
    "Cybersecurity Analyst":     ["SOC Analyst Intern","Security Analyst Intern","Pen Testing Intern"],
    "Web Developer":             ["Frontend Intern","Full Stack Intern","React Dev Intern"],
    "Cloud Engineer":            ["Cloud Intern","DevOps Intern","Infrastructure Intern"],
    "Software Developer":        ["SDE Intern","Backend Intern","Java/Python Dev Intern"],
}


CERTS = {
    "Machine Learning Engineer": ["Google ML Crash Course","Coursera Deep Learning Spec","AWS ML Specialty"],
    "Data Scientist":            ["IBM Data Science Professional","Google Data Analytics","Kaggle courses"],
    "Cybersecurity Analyst":     ["CompTIA Security+","CEH (EC-Council)","OSCP"],
    "Web Developer":             ["Meta Front-End Dev (Coursera)","freeCodeCamp","Google UX Design"],
    "Cloud Engineer":            ["AWS Cloud Practitioner","Google ACE","AZ-900 (Azure)"],
    "Software Developer":        ["Oracle Java SE","Meta Back-End Dev","Google Tech Dev Guide"],
}


SOFT_SKILLS = {
    "Machine Learning Engineer": ["Critical thinking","Problem decomposition","Research ability"],
    "Data Scientist":            ["Storytelling with data","Attention to detail","Business acumen"],
    "Cybersecurity Analyst":     ["Vigilance","Ethical mindset","Incident communication"],
    "Web Developer":             ["Creativity","User empathy","Collaboration"],
    "Cloud Engineer":            ["Systems thinking","Documentation","Cost awareness"],
    "Software Developer":        ["Logical thinking","Team communication","Code review etiquette"],
}


RESUME_TIPS = {
    "Machine Learning Engineer": [
        "Add Kaggle profile link with competition rank",
        "Mention model accuracy metrics in project descriptions",
        "List frameworks: TensorFlow, PyTorch, Scikit-learn",
        "Include a GitHub link with well-documented repos",
    ],
    "Data Scientist": [
        "Show dashboards/visualizations as project screenshots",
        "Quantify business impact (e.g., 'reduced churn by 15%')",
        "Mention SQL and Python proficiency prominently",
        "Add any Kaggle or real-world dataset analysis",
    ],
    "Cybersecurity Analyst": [
        "List tools: Wireshark, Kali Linux, Metasploit, Splunk",
        "Mention any CTF (Capture the Flag) participation",
        "Include bug bounty findings if any",
        "Add CompTIA or CEH certification clearly",
    ],
    "Web Developer": [
        "Show live project URLs (deployed on Vercel / Netlify)",
        "List tech stack clearly for each project",
        "Highlight responsive design and accessibility work",
        "Add GitHub contribution graph screenshot",
    ],
    "Cloud Engineer": [
        "List cloud services you've used (EC2, S3, Lambda, etc.)",
        "Show deployed architecture diagrams in portfolio",
        "Mention IaC tools: Terraform, CloudFormation",
        "Add AWS/Azure certification badge",
    ],
    "Software Developer": [
        "Highlight DSA problem-solving (LeetCode / HackerRank rank)",
        "Show GitHub contribution streak",
        "Mention system design projects",
        "Include open source contributions if any",
    ],
}


ALTERNATIVES = {
    "Machine Learning Engineer": ["Data Scientist","NLP Engineer","AI Research Scientist","MLOps Engineer"],
    "Data Scientist":            ["ML Engineer","Business Analyst","Data Engineer","BI Developer"],
    "Cybersecurity Analyst":     ["Ethical Hacker","Security Engineer","Cloud Security Architect","Forensics Analyst"],
    "Web Developer":             ["UI/UX Engineer","DevOps Engineer","Mobile Developer","Product Manager"],
    "Cloud Engineer":            ["DevOps Engineer","Site Reliability Engineer","Cloud Architect","Platform Engineer"],
    "Software Developer":        ["Backend Engineer","Mobile Developer","DevOps Engineer","Technical Lead"],
}


# ══════════════════════════════════════════════════════════════
# CAREER NAME NORMALIZATION
# Handles ANY spelling/casing/spacing your model or CSV uses,
# so it always matches the 6 dictionary keys above (no more
# missing "Cybersecurity" / "Web Development" lookups).
# ══════════════════════════════════════════════════════════════
def normalize_career(raw_career: str) -> str:
    s = str(raw_career).strip().lower()

    if "machine learning" in s or "ml engineer" in s or "ai engineer" in s or s == "ml":
        return "Machine Learning Engineer"
    elif "data scien" in s:
        return "Data Scientist"
    elif "cyber" in s or "security analyst" in s:
        return "Cybersecurity Analyst"
    elif "web" in s or "frontend" in s or "front-end" in s or "full stack" in s or "fullstack" in s:
        return "Web Developer"
    elif "cloud" in s:
        return "Cloud Engineer"
    elif "software" in s or s == "sde":
        return "Software Developer"
    else:
        return raw_career  # unknown — return as-is, fallbacks handle the rest


# ══════════════════════════════════════════════════════════════
# SIDEBAR — INPUT FORM
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎯 Career Mentor")
    st.markdown("Fill in your profile to get personalized recommendations.")
    st.divider()

    interest = st.selectbox("🔭 Your Interest", [
        "AI/ML","Data Science","Cybersecurity","Web Development","Cloud Computing","Software Development"
    ])
    primary_skill = st.selectbox("💻 Primary Skill", [
        "Python","JavaScript","Java","C++","SQL","Networking","Linux","R"
    ])
    coding_score  = st.slider("🖥️ Coding Score",      0, 100, 70)
    math_score    = st.slider("📐 Math Score",         0, 100, 65)
    comm_score    = st.slider("🗣️ Communication Score",0, 100, 72)
    projects      = st.number_input("📁 Projects Count",       0, 20, 3, step=1)
    certs         = st.number_input("🏅 Certifications Count", 0, 20, 1, step=1)
    github_score  = st.slider("🐙 GitHub Score",       0, 100, 60)
    personality   = st.selectbox("🧠 Personality Type", [
        "Analytical","Creative","Logical","Collaborative","Detail-Oriented"
    ])
    level = st.selectbox("🎓 Experience Level", ["Beginner","Intermediate","Advanced"])

    st.divider()
    predict_btn = st.button("🚀 Get My Career Recommendation", use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🎯 AI Career Mentor & Growth Planner</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Career Recommendation · Skill Gap Analysis · Learning Roadmap</div>', unsafe_allow_html=True)
st.divider()


# ══════════════════════════════════════════════════════════════
# PREDICTION LOGIC
# ══════════════════════════════════════════════════════════════
def predict_career(interest, primary_skill, coding_score, math_score,
                    comm_score, projects, certs, github_score, personality):
    """Build input row, encode it, and return predicted career + all class probabilities."""
    input_dict = {
        "Interest"            : interest,
        "Primary_Skill"       : primary_skill,
        "Coding_Score"        : coding_score,
        "Math_Score"          : math_score,
        "Communication_Score" : comm_score,
        "Projects_Count"      : projects,
        "Certifications_Count": certs,
        "GitHub_Score"        : github_score,
        "Personality_Type"    : personality,
    }

    row = {}
    for feat in feature_names:
        val = input_dict.get(feat, 0)
        if feat in label_enc and isinstance(val, str):
            try:
                val = label_enc[feat].transform([val])[0]
            except ValueError:
                val = 0
        row[feat] = val

    X_input = pd.DataFrame([row])[feature_names]

    # Probabilities from Random Forest
    probs    = rf.predict_proba(X_input)[0]
    classes  = label_enc["Career"].classes_
    prob_map = dict(zip(classes, probs * 100))

    # Final prediction from best model
    pred_idx = model.predict(X_input)[0]
    career   = label_enc["Career"].inverse_transform([pred_idx])[0]

    # Normalize career name so it matches the static dictionaries above
    career   = normalize_career(career)
    prob_map = {normalize_career(k): v for k, v in prob_map.items()}

    score = prob_map.get(career, 0)
    return career, score, prob_map


def skill_gap(user_skill, career):
    needed = SKILLS_NEEDED.get(career, [])
    has    = [user_skill]
    gaps   = [s for s in needed if s.lower() not in [h.lower() for h in has]]
    return needed, gaps


def readiness_score(coding, math, comm, projects, certs, github):
    """Simple weighted readiness metric out of 100."""
    score = (
        coding    * 0.25 +
        math      * 0.15 +
        comm      * 0.15 +
        min(projects, 10) / 10 * 100 * 0.20 +
        min(certs,    5)  / 5  * 100 * 0.15 +
        github    * 0.10
    )
    return round(score, 1)


# ══════════════════════════════════════════════════════════════
# MAIN RESULTS
# ══════════════════════════════════════════════════════════════
if predict_btn:
    if MODELS_LOADED:
        career, match_score, prob_map = predict_career(
            interest, primary_skill, coding_score, math_score,
            comm_score, projects, certs, github_score, personality
        )
    else:
        # Fallback: rule-based if models not trained yet
        mapping = {
            "AI/ML":                "Machine Learning Engineer",
            "Data Science":         "Data Scientist",
            "Cybersecurity":        "Cybersecurity Analyst",
            "Web Development":      "Web Developer",
            "Cloud Computing":      "Cloud Engineer",
            "Software Development": "Software Developer",
        }
        career      = mapping.get(interest, "Software Developer")
        match_score = 82.0
        prob_map    = {c: round(np.random.uniform(5, 25), 1) for c in mapping.values()}
        prob_map[career] = match_score

    needed_skills, gap_skills = skill_gap(primary_skill, career)
    intern_score = readiness_score(coding_score, math_score, comm_score, projects, certs, github_score)
    salary       = SALARY.get(career, ("₹5–10 LPA","₹12–25 LPA","High"))
    roadmap      = ROADMAPS.get(career, ROADMAPS["Software Developer"])
    internships  = INTERNSHIPS.get(career, [])
    cert_list    = CERTS.get(career, [])
    soft         = SOFT_SKILLS.get(career, [])
    resume_tips  = RESUME_TIPS.get(career, [])
    alts         = ALTERNATIVES.get(career, [])

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏆 Career Result",
        "📊 Skill Analysis",
        "🗓️ Roadmap",
        "💼 Internships",
        "💰 Salary & Growth",
        "📋 Resume Tips"
    ])

    # ══════════════════════════════════════════════════
    # TAB 1 — CAREER RESULT
    # ══════════════════════════════════════════════════
    with tab1:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(
                f"<div class='career-card'><h2 style='color:#3266ad;margin:0'>{career}</h2>"
                f"<p style='color:#555;margin:4px 0 0'>Best matched career for your profile</p></div>",
                unsafe_allow_html=True
            )
        with col2:
            st.metric("Match Score", f"{match_score:.1f}%", delta="AI Prediction")
        with col3:
            st.metric("Internship Readiness", f"{intern_score}%",
                      delta="Ready" if intern_score >= 60 else "Needs work")

        st.markdown("<div class='section-head'>All Career Match Scores</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 3.5))
        sorted_map = dict(sorted(prob_map.items(), key=lambda x: x[1]))
        colors = ["#3266ad" if k == career else "#a8c4e8" for k in sorted_map]
        bars = ax.barh(list(sorted_map.keys()), list(sorted_map.values()), color=colors, edgecolor="none")
        for bar, val in zip(bars, sorted_map.values()):
            ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va="center", fontsize=9)
        ax.set_xlabel("Match %")
        ax.set_title("Career Match Scores (Random Forest Probabilities)")
        ax.set_xlim(0, 110)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("<div class='section-head'>Alternative Career Paths</div>", unsafe_allow_html=True)
        if alts:
            cols = st.columns(min(len(alts), 4))
            for i, alt in enumerate(alts[:4]):
                cols[i % len(cols)].info(f"🔀 {alt}")
        else:
            st.info("Explore related fields based on your skills.")

    # ══════════════════════════════════════════════════
    # TAB 2 — SKILL ANALYSIS
    # ══════════════════════════════════════════════════
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='section-head'>Skills Needed for This Career</div>", unsafe_allow_html=True)
            for sk in needed_skills:
                has  = sk.lower() == primary_skill.lower()
                icon = "✅" if has else "❌"
                badge = '<span class="score-badge green-bg">You have this</span>' if has else '<span class="score-badge amber-bg">Learn this</span>'
                st.markdown(f"{icon} **{sk}**  {badge}", unsafe_allow_html=True)

            if gap_skills:
                st.markdown("<div class='section-head'>Skill Gap</div>", unsafe_allow_html=True)
                for g in gap_skills:
                    st.markdown(f"<div class='tip-box'>🔴 Missing: <strong>{g}</strong> — add to your learning plan</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='section-head'>Your Skill Radar</div>", unsafe_allow_html=True)
            categories = ["Coding", "Math", "Communication", "Projects\n(×10)", "Certs\n(×20)", "GitHub"]
            values     = [coding_score, math_score, comm_score,
                          min(projects*10, 100), min(certs*20, 100), github_score]
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            values += values[:1]
            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
            ax.fill(angles, values, color="#3266ad", alpha=0.25)
            ax.plot(angles, values, color="#3266ad", linewidth=2)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=9)
            ax.set_ylim(0, 100)
            ax.set_title("Skill Profile", pad=15)
            st.pyplot(fig)
            plt.close()

        st.markdown("<div class='section-head'>Recommended Certifications</div>", unsafe_allow_html=True)
        for c in cert_list:
            st.markdown(f"🏅 {c}")

        st.markdown("<div class='section-head'>Soft Skills to Develop</div>", unsafe_allow_html=True)
        for s in soft:
            st.markdown(f"<div class='tip-box'>🌟 {s}</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════
    # TAB 3 — ROADMAP
    # ══════════════════════════════════════════════════
    with tab3:
        st.markdown(f"<div class='section-head'>Learning Roadmap for {career}</div>", unsafe_allow_html=True)

        for i, (month, title, desc) in enumerate(roadmap, 1):
            col1, col2 = st.columns([1, 6])
            with col1:
                st.markdown(
                    f"<div style='text-align:center'>"
                    f"<div style='background:#3266ad;color:white;border-radius:50%;width:36px;height:36px;"
                    f"line-height:36px;margin:auto;font-weight:bold'>{i}</div>"
                    f"<div style='font-size:11px;color:#666;margin-top:4px'>{month}</div></div>",
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(f"**{title}**  \n{desc}")
            if i < len(roadmap):
                st.markdown("<hr style='margin:4px 0;border-color:#eee'>", unsafe_allow_html=True)

        st.markdown("<div class='section-head'>Monthly Progress Plan</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 3))
        months   = [r[0].split("–")[0].split(" ")[1] if " " in r[0] else r[0] for r in roadmap]
        progress = [round(100 * (i+1) / len(roadmap)) for i in range(len(roadmap))]
        ax.plot(months, progress, "o-", color="#3266ad", linewidth=2.5, markersize=8,
                markerfacecolor="white", markeredgewidth=2.5)
        ax.fill_between(months, progress, alpha=0.15, color="#3266ad")
        ax.set_ylim(0, 110)
        ax.set_ylabel("Completion %")
        ax.set_title("Projected Monthly Progress")
        ax.spines[["top","right"]].set_visible(False)
        for x, y, label in zip(months, progress, [r[1] for r in roadmap]):
            ax.annotate(f"{y}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ══════════════════════════════════════════════════
    # TAB 4 — INTERNSHIPS
    # ══════════════════════════════════════════════════
    with tab4:
        st.markdown("<div class='section-head'>Suitable Internship Domains</div>", unsafe_allow_html=True)
        for intern in internships:
            st.success(f"✅ {intern}")

        st.markdown("<div class='section-head'>Internship Readiness Score</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(4, 4))
            wedge_vals = [intern_score, 100 - intern_score]
            colors_pie = ["#3266ad", "#e8f0fb"]
            ax.pie(wedge_vals, colors=colors_pie, startangle=90,
                   wedgeprops=dict(width=0.45, edgecolor="white"))
            ax.text(0, 0, f"{intern_score}%", ha="center", va="center", fontsize=22, fontweight="bold", color="#3266ad")
            ax.set_title("Internship Readiness")
            st.pyplot(fig)
            plt.close()
        with col2:
            st.markdown("**Score Breakdown**")
            breakdown = {
                "Coding Score (25%)":   round(coding_score * 0.25, 1),
                "Math Score (15%)":     round(math_score * 0.15, 1),
                "Communication (15%)":  round(comm_score * 0.15, 1),
                "Projects (20%)":       round(min(projects/10, 1) * 100 * 0.20, 1),
                "Certifications (15%)": round(min(certs/5, 1) * 100 * 0.15, 1),
                "GitHub Score (10%)":   round(github_score * 0.10, 1),
            }
            for key, val in breakdown.items():
                st.write(f"• {key} → **{val}**")

        st.markdown("<div class='section-head'>Where to Find Internships</div>", unsafe_allow_html=True)
        platforms = {
            "🌐 InternShala": "https://internshala.com",
            "💼 LinkedIn":    "https://linkedin.com/jobs",
            "🔭 Naukri":      "https://naukri.com",
            "🌟 Unstop":      "https://unstop.com",
            "🏢 AngelList":   "https://angel.co",
        }
        cols = st.columns(len(platforms))
        for col, (name, url) in zip(cols, platforms.items()):
            col.markdown(f"[{name}]({url})")

    # ══════════════════════════════════════════════════
    # TAB 5 — SALARY & GROWTH
    # ══════════════════════════════════════════════════
    with tab5:
        entry, mid, growth = salary
        col1, col2, col3 = st.columns(3)
        col1.metric("Entry Level Salary", entry)
        col2.metric("Mid Level Salary",   mid)
        col3.metric("Growth Outlook",     growth)

        st.markdown("<div class='section-head'>Salary Comparison Across Careers</div>", unsafe_allow_html=True)
        careers_list = list(SALARY.keys())
        entry_min = [int(SALARY[c][0].replace("₹","").split("–")[0]) for c in careers_list]
        entry_max = [int(SALARY[c][0].replace("₹","").split("–")[1].replace(" LPA","")) for c in careers_list]
        x = np.arange(len(careers_list))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x - 0.2, entry_min, 0.35, label="Min Entry LPA", color="#a8c4e8", edgecolor="none")
        ax.bar(x + 0.2, entry_max, 0.35, label="Max Entry LPA", color="#3266ad", edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace(" ","\n") for c in careers_list], fontsize=8)
        ax.set_ylabel("LPA (Lakhs per Annum)")
        ax.set_title("Salary Range by Career (Entry Level)")
        ax.legend()
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ══════════════════════════════════════════════════
    # TAB 6 — RESUME TIPS
    # ══════════════════════════════════════════════════
    with tab6:
        st.markdown(f"<div class='section-head'>Resume Tips for {career}</div>", unsafe_allow_html=True)
        for tip in resume_tips:
            st.markdown(f"<div class='tip-box'>✏️ {tip}</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-head'>Resume Objective (one-liner)</div>", unsafe_allow_html=True)
        st.info(
            f"💡 'Motivated {level} enthusiast with hands-on experience in {primary_skill} and "
            f"{projects} projects, seeking a {career} internship to apply and grow my {interest} skills.'"
        )

        st.markdown("<div class='section-head'>Soft Skills Summary</div>", unsafe_allow_html=True)
        for s in soft:
            st.write(f"✔️ {s}")

else:
    # Welcome screen
    col1, col2, col3 = st.columns(3)
    col1.info("**Step 1** — Fill in your profile in the sidebar")
    col2.info("**Step 2** — Click 'Get My Career Recommendation'")
    col3.info("**Step 3** — Explore your personalized roadmap")

    st.markdown("### What you'll get")
    features = [
        ("🎯", "Career Recommendation", "AI-matched career based on your skills and interests"),
        ("📊", "Skill Gap Analysis",     "See exactly what skills you're missing"),
        ("🗓️", "Learning Roadmap",      "Step-by-step monthly plan to get there"),
        ("💼", "Internship Readiness",   "Score out of 100 with suggestions to improve"),
        ("💰", "Salary Information",     "Entry and mid-level salary ranges in India"),
        ("📋", "Resume Tips",            "Specific tips for your chosen career"),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        cols[i % 3].markdown(
            f"<div class='career-card'><b>{icon} {title}</b><br>"
            f"<small style='color:#666'>{desc}</small></div>",
            unsafe_allow_html=True
        )
