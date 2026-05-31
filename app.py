import streamlit as st
import pandas as pd
import pickle
import sqlite3
import hashlib
import uuid
import datetime
import plotly.graph_objects as go
from fpdf import FPDF


# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PulseIQ — Heart Disease AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  DATABASE SETUP
# ─────────────────────────────────────────────
DB_PATH = "pulseiq.db"

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            patient_id   TEXT PRIMARY KEY,
            username     TEXT UNIQUE NOT NULL,
            password     TEXT NOT NULL,
            full_name    TEXT NOT NULL,
            age          INTEGER,
            gender       TEXT,
            blood_group  TEXT,
            role         TEXT DEFAULT 'patient',
            created_at   TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            pred_id      TEXT PRIMARY KEY,
            patient_id   TEXT NOT NULL,
            username     TEXT NOT NULL,
            full_name    TEXT NOT NULL,
            age          INTEGER,
            sex          INTEGER,
            cp           INTEGER,
            trestbps     INTEGER,
            chol         INTEGER,
            thalch       INTEGER,
            exang        INTEGER,
            oldpeak      REAL,
            ca           INTEGER,
            thal         INTEGER,
            bmi          REAL,
            result       INTEGER,
            risk_pct     REAL,
            timestamp    TEXT,
            FOREIGN KEY (patient_id) REFERENCES users(patient_id)
        )
    """)

    doc_pw = hash_password("doctor123")
    c.execute("""
        INSERT OR IGNORE INTO users
        (patient_id, username, password, full_name, age, gender, blood_group, role, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("DR-000001", "doctor", doc_pw, "Dr. Admin", 40, "Male", "O+", "doctor",
          datetime.datetime.now().isoformat()))

    conn.commit()
    conn.close()

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def generate_patient_id() -> str:
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users WHERE role='patient'")
    count = c.fetchone()[0]
    conn.close()
    return f"PT-{str(count + 1).zfill(6)}"

def register_user(username, password, full_name, age, gender, blood_group):
    conn = get_conn()
    c = conn.cursor()
    try:
        pid = generate_patient_id()
        c.execute("""
            INSERT INTO users (patient_id, username, password, full_name, age, gender, blood_group, role, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'patient', ?)
        """, (pid, username, hash_password(password), full_name, age, gender, blood_group,
              datetime.datetime.now().isoformat()))
        conn.commit()
        return True, pid, ""
    except sqlite3.IntegrityError:
        return False, "", "Username already exists"
    finally:
        conn.close()

def login_user(username, password):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, hash_password(password)))
    row = c.fetchone()
    conn.close()
    if row:
        return dict(zip(['patient_id','username','password','full_name','age',
                         'gender','blood_group','role','created_at'], row))
    return None

def save_prediction(data: dict):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions
        (pred_id, patient_id, username, full_name, age, sex, cp, trestbps, chol,
         thalch, exang, oldpeak, ca, thal, bmi, result, risk_pct, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(uuid.uuid4()),
        data['patient_id'], data['username'], data['full_name'],
        data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'],
        data['thalch'], data['exang'], data['oldpeak'], data['ca'], data['thal'],
        data['bmi'], data['result'], data['risk_pct'],
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

def get_patient_history(patient_id):
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM predictions WHERE patient_id=? ORDER BY timestamp DESC",
        conn, params=(patient_id,))
    conn.close()
    return df

def get_all_predictions():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def get_all_patients():
    conn = get_conn()
    df = pd.read_sql_query("SELECT patient_id, username, full_name, age, gender, blood_group, created_at FROM users WHERE role='patient' ORDER BY created_at DESC", conn)
    conn.close()
    return df

init_db()

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pickle.load(open("heart_model.pkl", "rb"))

model = load_model()

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for k, v in {
    "logged_in": False,
    "user": None,
    "show_result": False,
    "last_prediction": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:        #04070f;
    --surface:   #080e1c;
    --surface2:  #0c1428;
    --border:    rgba(255,255,255,0.06);
    --border2:   rgba(255,255,255,0.1);
    --red:       #f43f5e;
    --red-dim:   rgba(244,63,94,0.12);
    --red-glow:  rgba(244,63,94,0.25);
    --green:     #10b981;
    --green-dim: rgba(16,185,129,0.1);
    --blue:      #3b82f6;
    --blue-dim:  rgba(59,130,246,0.1);
    --amber:     #f59e0b;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --subtle:    #1e293b;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: var(--bg) !important;
    font-family: 'Outfit', sans-serif;
    color: var(--text);
}

.stApp::before {
    content:''; position:fixed; top:-300px; right:-200px;
    width:700px; height:700px; pointer-events:none; z-index:0;
    background: radial-gradient(circle, rgba(244,63,94,0.05) 0%, transparent 65%);
}
.stApp::after {
    content:''; position:fixed; bottom:-300px; left:-200px;
    width:600px; height:600px; pointer-events:none; z-index:0;
    background: radial-gradient(circle, rgba(59,130,246,0.04) 0%, transparent 65%);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1280px; }

h1,h2,h3,h4,h5,h6 { font-family:'Outfit',sans-serif !important; color:#f1f5f9 !important; font-weight:700; }
p, li { color: var(--muted); line-height: 1.7; }
label { font-family:'Outfit',sans-serif !important; }

.hero { padding: 2.5rem 1rem 1.5rem; text-align: center; }
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--red-dim); border: 1px solid rgba(244,63,94,0.2);
    border-radius: 100px; padding: 5px 16px; margin-bottom: 20px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.15em;
    text-transform: uppercase; color: var(--red);
}
.hero-title {
    font-size: clamp(42px, 6vw, 76px); font-weight: 900;
    line-height: 1.0; letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 0%, #f43f5e 60%, #ff8c69 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub { color: var(--muted); font-size: 15px; margin-top: 14px; font-weight: 400; }

.card {
    background: linear-gradient(145deg, var(--surface), var(--surface2));
    border: 1px solid var(--border); border-radius: 20px;
    padding: 28px 30px; margin-bottom: 18px; position: relative; overflow: hidden;
    box-shadow: 0 8px 40px rgba(0,0,0,0.3);
}
.card::after {
    content:''; position:absolute; top:0; left:0; right:0; height:1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.07), transparent);
}
.card-red   { border-left: 3px solid var(--red); }
.card-green { border-left: 3px solid var(--green); }
.card-blue  { border-left: 3px solid var(--blue); }
.card-amber { border-left: 3px solid var(--amber); background: rgba(245,158,11,0.04); }

.auth-card {
    background: linear-gradient(160deg, #0a1020, #070d1a);
    border: 1px solid var(--border2); border-radius: 24px;
    padding: 48px 44px;
    box-shadow: 0 0 0 1px rgba(244,63,94,0.04), 0 30px 80px rgba(0,0,0,0.6),
                inset 0 1px 0 rgba(255,255,255,0.05);
    position: relative; overflow: hidden;
}
.auth-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:1px;
    background: linear-gradient(90deg, transparent, rgba(244,63,94,0.5), transparent);
}

.eyebrow {
    font-size: 10px; font-weight: 700; letter-spacing: 0.2em;
    text-transform: uppercase; color: var(--red); margin-bottom: 5px;
}
.section-title {
    font-size: 22px; font-weight: 800; color: #f1f5f9; margin-bottom: 14px; letter-spacing: -0.02em;
}
.page-title {
    font-size: clamp(26px, 3.5vw, 36px); font-weight: 900;
    color: #f8fafc; letter-spacing: -0.03em; margin-bottom: 6px;
}

.pill-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0; }
.pill {
    border-radius: 100px; padding: 5px 14px;
    font-size: 12px; font-weight: 600; border: 1px solid;
}
.pill-red   { background: var(--red-dim);   border-color: rgba(244,63,94,0.2);   color: #fda4af; }
.pill-green { background: var(--green-dim); border-color: rgba(16,185,129,0.2);  color: #6ee7b7; }
.pill-blue  { background: var(--blue-dim);  border-color: rgba(59,130,246,0.2);  color: #93c5fd; }
.pill-amber { background: rgba(245,158,11,0.1); border-color: rgba(245,158,11,0.2); color: #fcd34d; }

.result-box {
    border-radius: 20px; padding: 44px 24px 32px;
    text-align: center; margin-bottom: 20px; position: relative; overflow: hidden;
}
.result-danger { background: linear-gradient(160deg, #1c0710, #160810); border: 1px solid rgba(244,63,94,0.2); }
.result-safe   { background: linear-gradient(160deg, #061410, #050f0c); border: 1px solid rgba(16,185,129,0.2); }
.result-icon   { font-size: 60px; margin-bottom: 10px; }
.result-tag    { font-size: 11px; font-weight: 700; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 8px; }
.result-tag-red   { color: var(--red); }
.result-tag-green { color: var(--green); }
.result-headline  { font-family: 'Outfit',sans-serif; font-size: clamp(22px,3.5vw,38px); font-weight: 900; letter-spacing: -0.02em; }
.result-headline-red   { color: #fda4af; }
.result-headline-green { color: #6ee7b7; }

.stats-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 24px; }
.stat-box {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; padding: 20px 22px;
    display: flex; flex-direction: column; gap: 4px;
}
.stat-value { font-size: 32px; font-weight: 900; color: #f1f5f9; letter-spacing: -0.03em; }
.stat-label { font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
.stat-delta { font-size: 12px; font-weight: 600; }
.delta-up   { color: var(--red); }
.delta-ok   { color: var(--green); }

.med-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px,1fr)); gap: 14px; margin-top: 14px; }
.med-card {
    background: rgba(244,63,94,0.05); border: 1px solid rgba(244,63,94,0.1);
    border-radius: 14px; padding: 20px;
}
.med-icon  { font-size: 26px; margin-bottom: 8px; }
.med-name  { font-size: 15px; font-weight: 800; color: #fda4af; margin-bottom: 4px; }
.med-class { font-size: 10px; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
.med-desc  { font-size: 12px; color: #475569; line-height: 1.5; }

.feat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px,1fr)); gap: 12px; margin-top: 14px; }
.feat-card {
    background: rgba(255,255,255,0.02); border: 1px solid var(--border);
    border-radius: 14px; padding: 20px 16px; text-align: center;
}
.feat-icon  { font-size: 28px; margin-bottom: 10px; }
.feat-label { font-size: 13px; font-weight: 700; color: #e2e8f0; }

.pid-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--red-dim); border: 1px solid rgba(244,63,94,0.2);
    border-radius: 10px; padding: 10px 18px; margin: 14px 0;
    font-family: 'JetBrains Mono', monospace; font-size: 18px;
    font-weight: 500; color: #fda4af; letter-spacing: 0.05em;
}

[data-testid="stSidebar"] {
    background: #050a16 !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--muted) !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }

.sb-brand { font-size: 22px; font-weight: 900; letter-spacing: -0.02em; color: #f8fafc; margin-bottom: 2px; }
.sb-brand span { color: var(--red); }
.sb-tag   { font-size: 10px; letter-spacing: 0.15em; text-transform: uppercase; color: #1e293b; margin-bottom: 24px; }
.sb-user  {
    background: var(--red-dim); border: 1px solid rgba(244,63,94,0.15);
    border-radius: 12px; padding: 14px 16px; margin-bottom: 20px;
}
.sb-role  { font-size: 9px; letter-spacing: 0.15em; text-transform: uppercase; color: #475569; }
.sb-name  { font-size: 15px; font-weight: 800; color: #fda4af !important; }
.sb-pid   { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #475569 !important; margin-top: 3px; }

.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #0a1020 !important; border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important; color: var(--text) !important;
    font-family: 'Outfit',sans-serif !important; font-size: 15px !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: rgba(244,63,94,0.45) !important;
    box-shadow: 0 0 0 3px rgba(244,63,94,0.07) !important;
}
.stSelectbox > div > div,
.stSelectbox > div > div > div {
    background: #0a1020 !important; border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important; color: var(--text) !important;
}
.stSlider > div > div > div > div { background: var(--red) !important; }

label, .stSelectbox label, .stTextInput label, .stNumberInput label,
.stSlider label, .stRadio label {
    color: var(--muted) !important; font-size: 11px !important;
    font-weight: 600 !important; letter-spacing: 0.12em !important;
    text-transform: uppercase !important; font-family: 'Outfit',sans-serif !important;
}

div.stButton > button {
    background: linear-gradient(135deg, #e11d48, #f43f5e) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    height: 50px !important; width: 100% !important;
    font-size: 15px !important; font-weight: 700 !important;
    font-family: 'Outfit',sans-serif !important; letter-spacing: 0.01em !important;
    box-shadow: 0 4px 24px rgba(244,63,94,0.3) !important;
    transition: all 0.2s !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(244,63,94,0.4) !important;
    background: linear-gradient(135deg, #be123c, #e11d48) !important;
}
div.stButton > button:active { transform: translateY(0) !important; }

.stRadio > div {
    background: #0a1020; border: 1px solid var(--border); border-radius: 12px; padding: 6px; gap: 4px;
}

.stDataFrame { border-radius: 14px; overflow: hidden; }
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 14px !important; }

.stSuccess { background: rgba(16,185,129,0.08) !important; border: 1px solid rgba(16,185,129,0.2) !important; border-radius: 12px !important; }
.stError   { background: rgba(244,63,94,0.08) !important; border: 1px solid rgba(244,63,94,0.2) !important; border-radius: 12px !important; }
.stWarning { background: rgba(245,158,11,0.08) !important; border: 1px solid rgba(245,158,11,0.2) !important; border-radius: 12px !important; }
.stInfo    { background: rgba(59,130,246,0.08) !important; border: 1px solid rgba(59,130,246,0.2) !important; border-radius: 12px !important; }

.input-panel {
    background: rgba(255,255,255,0.015); border: 1px solid var(--border);
    border-radius: 16px; padding: 24px 26px; margin-bottom: 16px;
}

hr { border-color: var(--border) !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--subtle); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit", color="#64748b"),
    margin=dict(t=60, b=30, l=20, r=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", showline=False, tickfont=dict(color="#475569")),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", showline=False, tickfont=dict(color="#475569")),
    legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
)

def make_gauge(risk_pct):
    color = "#f43f5e" if risk_pct >= 50 else "#f59e0b" if risk_pct >= 30 else "#10b981"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number={"suffix": "%", "font": {"size": 42, "family": "Outfit", "color": "#f1f5f9"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#334155",
                     "tickfont": {"color": "#475569", "family": "Outfit"}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "rgba(16,185,129,0.1)"},
                {"range": [30, 60], "color": "rgba(245,158,11,0.1)"},
                {"range": [60,100], "color": "rgba(244,63,94,0.1)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": risk_pct},
        },
        title={"text": "CARDIAC RISK SCORE", "font": {"size": 11, "family": "Outfit", "color": "#64748b"}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=260, margin=dict(t=40, b=10, l=30, r=30),
    )
    return fig


def generate_pdf(pred: dict, user: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # header
    pdf.set_fill_color(10, 14, 28)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(244, 63, 94)
    pdf.cell(0, 15, "", ln=True)
    pdf.cell(0, 12, "PulseIQ", ln=False, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.ln(6)
    pdf.cell(0, 8, "Heart Disease AI Prediction Report", ln=True, align="C")
    pdf.ln(10)

    # patient info box
    pdf.set_fill_color(15, 22, 40)
    pdf.set_draw_color(30, 41, 59)
    pdf.rect(10, pdf.get_y(), 190, 42, 'FD')
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.set_x(14)
    pdf.cell(0, 8, "PATIENT INFORMATION", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(226, 232, 240)

    info = [
        ("Patient ID",    user.get("patient_id", "N/A")),
        ("Full Name",     user.get("full_name", "N/A")),
        ("Age",           str(pred.get("age", "N/A"))),
        ("Gender",        "Male" if pred.get("sex") == 1 else "Female"),
        ("Blood Group",   user.get("blood_group", "N/A")),
        ("Report Date",   pred.get("timestamp", "N/A")),
    ]
    for i in range(0, len(info), 2):
        pdf.set_x(14)
        label1, val1 = info[i]
        label2, val2 = info[i+1] if i+1 < len(info) else ("", "")
        pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(100, 116, 139)
        pdf.cell(45, 6, label1 + ":", ln=False)
        pdf.set_font("Helvetica", "", 10); pdf.set_text_color(226, 232, 240)
        pdf.cell(50, 6, str(val1), ln=False)
        pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(100, 116, 139)
        pdf.cell(45, 6, label2 + ":" if label2 else "", ln=False)
        pdf.set_font("Helvetica", "", 10); pdf.set_text_color(226, 232, 240)
        pdf.cell(50, 6, str(val2), ln=True)
    pdf.ln(6)

    # result — FIX: use only ASCII-safe characters (no Unicode symbols)
    risk = pred.get("risk_pct", 0)
    res  = pred.get("result", 0)
    if res == 1:
        pdf.set_fill_color(30, 8, 18); pdf.set_draw_color(244, 63, 94)
        pdf.set_text_color(253, 164, 175)
    else:
        pdf.set_fill_color(6, 20, 16); pdf.set_draw_color(16, 185, 129)
        pdf.set_text_color(110, 231, 183)
    pdf.rect(10, pdf.get_y(), 190, 22, 'FD')
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_x(14)
    # ASCII-safe verdict strings (no special Unicode chars)
    verdict = "!! Heart Disease Risk DETECTED" if res == 1 else ">> No Significant Risk Detected"
    pdf.cell(0, 10, verdict, ln=False, align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Risk Score: {risk:.1f}%", ln=True, align="C")
    pdf.ln(8)

    # clinical params
    pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 8, "CLINICAL PARAMETERS", ln=True)
    pdf.set_draw_color(30, 41, 59)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    cp_map = {0:"Typical Angina", 1:"Atypical Angina", 2:"Non-anginal Pain", 3:"Asymptomatic"}
    thal_map = {0:"Normal", 1:"Fixed Defect", 2:"Reversible Defect"}
    params = [
        ("Age",                str(pred.get("age", ""))),
        ("Blood Pressure",     f"{pred.get('trestbps','')} mmHg"),
        ("Cholesterol",        f"{pred.get('chol','')} mg/dL"),
        ("Max Heart Rate",     f"{pred.get('thalch','')} bpm"),
        ("Oldpeak (ST Dep.)",  str(pred.get("oldpeak",""))),
        ("Major Vessels (CA)", str(pred.get("ca",""))),
        ("Chest Pain Type",    cp_map.get(pred.get("cp",0), "")),
        ("Exercise Angina",    "Yes" if pred.get("exang") == 1 else "No"),
        ("Thalassemia",        thal_map.get(pred.get("thal",0), "")),
        ("BMI",                f"{pred.get('bmi', 0):.1f}"),
    ]
    for i in range(0, len(params), 2):
        pdf.set_x(14)
        l1, v1 = params[i]
        l2, v2 = params[i+1] if i+1 < len(params) else ("","")
        pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(100, 116, 139)
        pdf.cell(50, 7, l1 + ":", ln=False)
        pdf.set_font("Helvetica", "", 10); pdf.set_text_color(226, 232, 240)
        pdf.cell(45, 7, v1, ln=False)
        pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(100, 116, 139)
        pdf.cell(50, 7, l2 + ":" if l2 else "", ln=False)
        pdf.set_font("Helvetica", "", 10); pdf.set_text_color(226, 232, 240)
        pdf.cell(45, 7, v2, ln=True)
    pdf.ln(6)

    # disclaimer
    pdf.set_fill_color(12, 18, 32); pdf.set_draw_color(30, 41, 59)
    pdf.rect(10, pdf.get_y(), 190, 20, 'FD')
    pdf.set_font("Helvetica", "I", 8); pdf.set_text_color(71, 85, 105)
    pdf.set_x(14)
    pdf.multi_cell(182, 5,
        "DISCLAIMER: This report is generated by an AI model (Random Forest, ~86% accuracy on UCI Heart Disease dataset) "
        "and is intended for educational and screening purposes only. It does not constitute medical advice. "
        "Please consult a qualified cardiologist for diagnosis and treatment.")

    return bytes(pdf.output())


# ══════════════════════════════════════════════
#  AUTH SCREEN
# ══════════════════════════════════════════════
if not st.session_state.logged_in:

    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">🫀 AI-Powered Cardiac Screening</div>
        <div class="hero-title">PulseIQ</div>
        <p class="hero-sub">Intelligent heart disease risk assessment — powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.2, 1])

    with col:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)

        tab = st.radio("", ["🔐  Login", "📝  Register"], horizontal=True, label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)

        if "Login" in tab:
            st.markdown('<div class="eyebrow">Access Your Account</div>', unsafe_allow_html=True)
            u = st.text_input("Username", placeholder="Enter your username")
            p = st.text_input("Password", type="password", placeholder="Enter your password")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Sign In →"):
                if u == "" or p == "":
                    st.error("Please fill in all fields.")
                else:
                    user = login_user(u, p)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user = user
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

        

        else:
            st.markdown('<div class="eyebrow">Create Patient Account</div>', unsafe_allow_html=True)

            rc1, rc2 = st.columns(2)
            with rc1:
                full_name = st.text_input("Full Name", placeholder="John Smith")
                new_user  = st.text_input("Username", placeholder="Choose a username")
                new_pass  = st.text_input("Password", type="password", placeholder="Min 6 chars")
            with rc2:
                reg_age   = st.number_input("Age", min_value=1, max_value=110, value=30)
                reg_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                reg_bg    = st.selectbox("Blood Group", ["A+","A-","B+","B-","AB+","AB-","O+","O-","Unknown"])

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account →"):
                if not all([full_name, new_user, new_pass]):
                    st.error("Please fill in all fields.")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    ok, pid, err = register_user(new_user, new_pass, full_name, reg_age, reg_gender, reg_bg)
                    if ok:
                        st.success(f"Account created! Your Patient ID: **{pid}**")
                        st.info("Please switch to Login to sign in.")
                    else:
                        st.error(err)

        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════
else:
    user = st.session_state.user
    is_doctor = user["role"] == "doctor"

    with st.sidebar:
        st.markdown("""
        <div class="sb-brand">Pulse<span>IQ</span></div>
        <div class="sb-tag">Heart Disease AI Platform</div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sb-user">
            <div class="sb-role">{"🩺 DOCTOR" if is_doctor else "👤 PATIENT"}</div>
            <div class="sb-name">{user['full_name']}</div>
            <div class="sb-pid">{user['patient_id']}</div>
        </div>
        """, unsafe_allow_html=True)

        if is_doctor:
            nav_options = ["🩺  Prediction", "📋  All Records", "👥  Patients", "📊  Analytics", "💊  Medicines", "ℹ️  About", "🚪  Logout"]
        else:
            nav_options = ["🩺  Prediction", "📋  My History", "📊  Analytics", "💊  Medicines", "ℹ️  About", "🚪  Logout"]

        menu = st.radio("", nav_options, label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size:10px;color:#1e293b;text-align:center;line-height:1.8;">
        Model: Random Forest<br>
        Accuracy: ~86%<br>
        Dataset: UCI Heart Disease<br><br>
        For educational use only.<br>Consult a physician.
        </p>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    #  PREDICTION
    # ══════════════════════════════════════════
    if "Prediction" in menu:

        if not st.session_state.show_result:

            st.markdown("""
            <div class="eyebrow">Cardiac Analysis</div>
            <div class="page-title">Heart Disease Prediction</div>
            <p style="margin-bottom:24px;">Enter clinical parameters for AI-powered risk assessment.</p>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 1, 1], gap="large")

            with col1:
                st.markdown('<div class="input-panel">', unsafe_allow_html=True)
                st.markdown('<div class="eyebrow">Demographics</div>', unsafe_allow_html=True)
                # FIX: All default values set to minimum/zero so form opens blank
                age    = st.number_input("Age (years)", min_value=1, max_value=120, value=1)
                sex_opt = st.selectbox("Biological Sex", ["Male", "Female"], index=0)
                sex = 1 if sex_opt == "Male" else 0
                height = st.number_input("Height (cm)", min_value=100, max_value=250, value=100)
                weight = st.number_input("Weight (kg)", min_value=20, max_value=300, value=20)
                bmi = round(weight / ((height / 100) ** 2), 1)
                bmi_cat = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
                st.markdown(f"""
                <div style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.15);
                border-radius:10px;padding:10px 14px;margin-top:4px;">
                    <span style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#64748b;">
                    Calculated BMI</span><br>
                    <span style="font-size:20px;font-weight:900;color:#93c5fd;">{bmi}</span>
                    <span style="font-size:12px;color:#3b82f6;margin-left:6px;">{bmi_cat}</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="input-panel">', unsafe_allow_html=True)
                st.markdown('<div class="eyebrow">Vitals</div>', unsafe_allow_html=True)
                bp      = st.number_input("Resting BP (mmHg)", min_value=60, max_value=250, value=60)
                chol    = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=100)
                thalach = st.number_input("Max Heart Rate (bpm)", min_value=60, max_value=250, value=60)
                oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="input-panel">', unsafe_allow_html=True)
                st.markdown('<div class="eyebrow">Clinical Tests</div>', unsafe_allow_html=True)
                cp = st.select_slider("Chest Pain Type", options=[0,1,2,3],
                    format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",
                                           2:"Non-anginal",3:"Asymptomatic"}[x])
                exang_opt = st.selectbox("Exercise-induced Angina", ["No","Yes"])
                exang = 1 if exang_opt == "Yes" else 0
                ca   = st.slider("Major Vessels (0–3)", 0, 3, 0)
                thal = st.select_slider("Thalassemia", options=[0,1,2],
                    format_func=lambda x: {0:"Normal",1:"Fixed Defect",2:"Reversible Defect"}[x])
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            _, btn_col, _ = st.columns([1.5, 2, 1.5])
            with btn_col:
                if st.button("🔍  Analyse Cardiac Risk"):
                    inp = pd.DataFrame([[age, sex, cp, bp, chol, thalach, exang, oldpeak, ca, thal]],
                        columns=['age','sex','cp','trestbps','chol','thalch','exang','oldpeak','ca','thal'])
                    pred    = model.predict(inp)[0]
                    prob    = model.predict_proba(inp)[0]
                    risk_pct = round(float(prob[1]) * 100, 1)

                    pred_data = {
                        "patient_id": user["patient_id"], "username": user["username"],
                        "full_name": user["full_name"], "age": age, "sex": sex, "cp": cp,
                        "trestbps": bp, "chol": chol, "thalch": thalach, "exang": exang,
                        "oldpeak": oldpeak, "ca": ca, "thal": thal, "bmi": bmi,
                        "result": int(pred), "risk_pct": risk_pct,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    save_prediction(pred_data)
                    st.session_state.last_prediction = {**pred_data, **{"user": user}}
                    st.session_state.show_result = True
                    st.rerun()

        else:
            pred = st.session_state.last_prediction
            risk = pred["risk_pct"]
            res  = pred["result"]

            st.markdown("""
            <div class="eyebrow">Analysis Complete</div>
            <div class="page-title">Prediction Result</div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            left, right = st.columns([1, 1], gap="large")

            with left:
                if res == 1:
                    st.markdown(f"""
                    <div class="result-box result-danger">
                        <div class="result-icon">🫀</div>
                        <div class="result-tag result-tag-red">Risk Detected</div>
                        <div class="result-headline result-headline-red">Heart Disease Risk Present</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box result-safe">
                        <div class="result-icon">💚</div>
                        <div class="result-tag result-tag-green">All Clear</div>
                        <div class="result-headline result-headline-green">No Significant Risk Detected</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.plotly_chart(make_gauge(risk), use_container_width=True)

            with right:
                st.markdown(f"""
                <div style="display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap;">
                    <div class="pill {'pill-red' if res==1 else 'pill-green'}">
                        Risk: {risk:.1f}%
                    </div>
                    <div class="pill pill-blue">BMI: {pred['bmi']}</div>
                    <div class="pill pill-amber">Age: {pred['age']}</div>
                </div>
                """, unsafe_allow_html=True)

                if res == 1:
                    st.markdown("""
                    <div class="card card-red">
                        <div class="eyebrow">⚠️ Health Advisory</div>
                        <div class="section-title">What This Means</div>
                        <p>Clinical indicators suggest elevated cardiovascular risk. This is <strong style="color:#fda4af;">not a diagnosis</strong> — a qualified cardiologist should perform a complete evaluation.</p>
                    </div>
                    <div class="card">
                        <div class="eyebrow">Recommended Actions</div>
                        <div class="section-title">Next Steps</div>
                        <ul style="padding-left:18px;line-height:2.2;">
                            <li>Book an appointment with a cardiologist immediately</li>
                            <li>Reduce sodium, saturated fats &amp; processed foods</li>
                            <li>Start a supervised cardiac exercise programme</li>
                            <li>Monitor BP and cholesterol every 2 weeks</li>
                            <li>Avoid smoking, excessive alcohol and chronic stress</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="card card-green">
                        <div class="eyebrow">🌿 Positive Result</div>
                        <div class="section-title">Great News!</div>
                        <p>No major cardiovascular risk markers detected. Continue your healthy routine and keep up with annual cardiac check-ups.</p>
                    </div>
                    <div class="card">
                        <div class="eyebrow">Maintenance Tips</div>
                        <div class="section-title">Keep It Up</div>
                        <ul style="padding-left:18px;line-height:2.2;">
                            <li>Stay active — aim for 150 min/week of moderate cardio</li>
                            <li>Eat a fibre-rich, heart-healthy diet</li>
                            <li>Stay well-hydrated throughout the day</li>
                            <li>Prioritise 7–8 hours of quality sleep nightly</li>
                            <li>Schedule an annual cardiac screening</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                pdf_bytes = generate_pdf(pred, user)
                st.download_button(
                    label="📄  Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"PulseIQ_{user['patient_id']}_{datetime.date.today()}.pdf",
                    mime="application/pdf",
                )

            st.markdown("<br>", unsafe_allow_html=True)
            _, back_col, _ = st.columns([1.5, 2, 1.5])
            with back_col:
                if st.button("← Run Another Analysis"):
                    st.session_state.show_result = False
                    st.rerun()

    # ══════════════════════════════════════════
    #  PATIENT — MY HISTORY
    # ══════════════════════════════════════════
    elif "My History" in menu:

        st.markdown("""
        <div class="eyebrow">Personal Records</div>
        <div class="page-title">My Prediction History</div>
        <p style="margin-bottom:24px;">All your past cardiac risk assessments, most recent first.</p>
        """, unsafe_allow_html=True)

        df = get_patient_history(user["patient_id"])

        if df.empty:
            st.info("No predictions yet. Head to the Prediction page to run your first analysis.")
        else:
            total = len(df)
            at_risk = int((df["result"] == 1).sum())
            avg_risk = df["risk_pct"].mean()
            last_date = df["timestamp"].iloc[0][:10]

            st.markdown(f"""
            <div class="stats-grid" style="grid-template-columns:repeat(4,1fr);">
                <div class="stat-box">
                    <div class="stat-value">{total}</div>
                    <div class="stat-label">Total Scans</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" style="color:#f43f5e;">{at_risk}</div>
                    <div class="stat-label">At-Risk Results</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{avg_risk:.1f}%</div>
                    <div class="stat-label">Avg Risk Score</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" style="font-size:18px;">{last_date}</div>
                    <div class="stat-label">Last Scan</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            display_df = df[["timestamp","age","risk_pct","result","bmi"]].copy()
            display_df.columns = ["Timestamp","Age","Risk %","Result","BMI"]
            display_df["Result"] = display_df["Result"].map({1:"⚠ At Risk", 0:"✓ Safe"})
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            if len(df) > 1:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=df["timestamp"], y=df["risk_pct"],
                    mode="lines+markers",
                    line=dict(color="#f43f5e", width=2),
                    marker=dict(color="#f43f5e", size=7),
                    fill="tozeroy", fillcolor="rgba(244,63,94,0.07)",
                    name="Risk %"
                ))
                fig2.update_layout(
                    **CHART_LAYOUT,
                    title=dict(text="<b>Risk Score Over Time</b>",
                               font=dict(color="#f1f5f9", size=15, family="Outfit"), x=0.02),
                    height=320,
                )
                st.plotly_chart(fig2, use_container_width=True)

    # ══════════════════════════════════════════
    #  DOCTOR — ALL RECORDS
    # ══════════════════════════════════════════
    elif "All Records" in menu and is_doctor:

        st.markdown("""
        <div class="eyebrow">Doctor Dashboard</div>
        <div class="page-title">All Patient Records</div>
        <p style="margin-bottom:24px;">Complete prediction history across all patients.</p>
        """, unsafe_allow_html=True)

        df = get_all_predictions()

        if df.empty:
            st.info("No predictions recorded yet.")
        else:
            total = len(df)
            at_risk = int((df["result"] == 1).sum())
            avg_risk = df["risk_pct"].mean()
            patients = df["patient_id"].nunique()

            st.markdown(f"""
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-value">{total}</div>
                    <div class="stat-label">Total Predictions</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" style="color:#f43f5e;">{at_risk}</div>
                    <div class="stat-label">At-Risk Cases</div>
                    <div class="stat-delta delta-up">{at_risk/total*100:.1f}% of total</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{avg_risk:.1f}%</div>
                    <div class="stat-label">Avg Risk Score</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{patients}</div>
                    <div class="stat-label">Unique Patients</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            display_df = df[["timestamp","full_name","patient_id","age","risk_pct","result","bmi"]].copy()
            display_df.columns = ["Timestamp","Patient","ID","Age","Risk %","Result","BMI"]
            display_df["Result"] = display_df["Result"].map({1:"⚠ At Risk", 0:"✓ Safe"})
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════
    #  DOCTOR — PATIENTS
    # ══════════════════════════════════════════
    elif "Patients" in menu and is_doctor:

        st.markdown("""
        <div class="eyebrow">Patient Management</div>
        <div class="page-title">Registered Patients</div>
        <p style="margin-bottom:24px;">All registered patient accounts.</p>
        """, unsafe_allow_html=True)

        df = get_all_patients()
        if df.empty:
            st.info("No patients registered yet.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════
    #  ANALYTICS
    # ══════════════════════════════════════════
    elif "Analytics" in menu:

        st.markdown("""
        <div class="eyebrow">Data Insights</div>
        <div class="page-title">Analytics Dashboard</div>
        <p style="margin-bottom:24px;">Visual breakdown of prediction data and risk patterns.</p>
        """, unsafe_allow_html=True)

        df = get_all_predictions() if is_doctor else get_patient_history(user["patient_id"])

        if df.empty:
            st.info("No data available yet. Run a prediction first.")
        else:
            RED   = "#f43f5e"
            GREEN = "#10b981"
            BLUE  = "#3b82f6"
            AMBER = "#f59e0b"

            r1c1, r1c2 = st.columns(2, gap="large")

            with r1c1:
                counts = df["result"].value_counts()
                fig1 = go.Figure(go.Pie(
                    labels=["At Risk" if l == 1 else "Safe" for l in counts.index],
                    values=counts.values,
                    hole=0.6,
                    marker=dict(colors=[RED, GREEN], line=dict(color="#04070f", width=3)),
                    textfont=dict(family="Outfit", size=13),
                ))
                fig1.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(font=dict(color="#94a3b8", family="Outfit"), bgcolor="rgba(0,0,0,0)"),
                    title=dict(text="<b>Risk Distribution</b>", font=dict(color="#f1f5f9", size=15, family="Outfit"), x=0.5),
                    height=320, margin=dict(t=60, b=20, l=20, r=20),
                )
                st.plotly_chart(fig1, use_container_width=True)

            with r1c2:
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(
                    x=df["risk_pct"], nbinsx=20,
                    marker=dict(color=RED, opacity=0.8, line=dict(color="#04070f", width=1)),
                    name="Risk %"
                ))
                fig2.update_layout(
                    **CHART_LAYOUT,
                    title=dict(text="<b>Risk Score Distribution</b>",
                               font=dict(color="#f1f5f9", size=15, family="Outfit"), x=0.5),
                    height=320,
                    bargap=0.05,
                )
                st.plotly_chart(fig2, use_container_width=True)

            r2c1, r2c2 = st.columns(2, gap="large")

            with r2c1:
                fig3 = go.Figure()
                fig3.add_trace(go.Box(
                    y=df[df["result"]==1]["age"], name="At Risk",
                    marker_color=RED, line_color=RED, fillcolor="rgba(244,63,94,0.1)"
                ))
                fig3.add_trace(go.Box(
                    y=df[df["result"]==0]["age"], name="Safe",
                    marker_color=GREEN, line_color=GREEN, fillcolor="rgba(16,185,129,0.1)"
                ))
                fig3.update_layout(
                    **CHART_LAYOUT,
                    title=dict(text="<b>Age by Risk Category</b>",
                               font=dict(color="#f1f5f9", size=15, family="Outfit"), x=0.5),
                    height=340,
                )
                st.plotly_chart(fig3, use_container_width=True)

            with r2c2:
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=df["age"], y=df["risk_pct"],
                    mode="markers",
                    marker=dict(
                        color=df["result"].map({1: RED, 0: GREEN}),
                        size=9, opacity=0.8,
                        line=dict(color="#04070f", width=1)
                    ),
                    name="Patients"
                ))
                fig4.update_layout(
                    **CHART_LAYOUT,
                    title=dict(text="<b>Age vs Risk Score</b>",
                               font=dict(color="#f1f5f9", size=15, family="Outfit"), x=0.5),
                    height=340,
                    xaxis_title="Age", yaxis_title="Risk %",
                )
                st.plotly_chart(fig4, use_container_width=True)

            categories = ["Hypertension", "High Chol.", "Diabetes", "Obesity", "Smoking", "Inactivity"]
            values = [72, 65, 48, 54, 59, 63]
            fig5 = go.Figure(go.Scatterpolar(
                r=values + [values[0]], theta=categories + [categories[0]],
                fill="toself", fillcolor="rgba(244,63,94,0.1)",
                line=dict(color=RED, width=2),
                marker=dict(color=RED, size=7),
            ))
            fig5.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                polar=dict(bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0,100], gridcolor="rgba(255,255,255,0.05)",
                                    tickfont=dict(color="#475569", family="Outfit")),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.05)",
                                     tickfont=dict(color="#94a3b8", family="Outfit", size=12))),
                title=dict(text="<b>Top Risk Factor Prevalence (%)</b>", font=dict(color="#f1f5f9", size=15, family="Outfit"), x=0.5),
                height=380, margin=dict(t=60, b=30),
            )
            st.plotly_chart(fig5, use_container_width=True)
            

    # ══════════════════════════════════════════
    #  MEDICINES
    # ══════════════════════════════════════════
    elif "Medicines" in menu:

        st.markdown("""
        <div class="eyebrow">Pharmacology Reference</div>
        <div class="page-title">Cardiac Medicines</div>
        <p style="margin-bottom:24px;">Common medications used in cardiovascular care. Always follow your doctor's prescription.</p>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
        <div class="eyebrow">Common Prescriptions</div>
        <div class="section-title">Frequently Used Heart Medications</div>
        <div class="med-grid">
            <div class="med-card">
                <div class="med-icon">💊</div>
                <div class="med-name">Aspirin</div>
                <div class="med-class">Antiplatelet</div>
                <div class="med-desc">Reduces blood clot formation in coronary arteries. Widely used in secondary prevention.</div>
            </div>
            <div class="med-card">
                <div class="med-icon">💊</div>
                <div class="med-name">Atorvastatin</div>
                <div class="med-class">Statin</div>
                <div class="med-desc">Lowers LDL cholesterol and reduces arterial plaque build-up significantly.</div>
            </div>
            <div class="med-card">
                <div class="med-icon">💊</div>
                <div class="med-name">Clopidogrel</div>
                <div class="med-class">Antiplatelet</div>
                <div class="med-desc">Prevents thrombosis post-stent placement or angioplasty procedures.</div>
            </div>
            <div class="med-card">
                <div class="med-icon">💊</div>
                <div class="med-name">Metoprolol</div>
                <div class="med-class">Beta Blocker</div>
                <div class="med-desc">Reduces heart rate and BP; manages angina, arrhythmia, and heart failure.</div>
            </div>
            <div class="med-card">
                <div class="med-icon">💊</div>
                <div class="med-name">Ramipril</div>
                <div class="med-class">ACE Inhibitor</div>
                <div class="med-desc">Relaxes blood vessels; used in heart failure, post-MI, and hypertension.</div>
            </div>
            <div class="med-card">
                <div class="med-icon">💊</div>
                <div class="med-name">Nitroglycerin</div>
                <div class="med-class">Vasodilator</div>
                <div class="med-desc">Fast-acting relief for acute chest pain and angina episodes.</div>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card card-amber">
            <div class="eyebrow" style="color:#f59e0b;">⚠️ Important Disclaimer</div>
            <p style="color:#fcd34d;">The medications above are general references only. Dosage, suitability,
            and drug interactions vary per individual. <strong>Never self-medicate.</strong>
            Always consult a qualified cardiologist before starting, changing, or stopping any medication.</p>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    #  ABOUT
    # ══════════════════════════════════════════
    elif "About" in menu:

        st.markdown("""
        <div class="eyebrow">Project Information</div>
        <div class="page-title">About PulseIQ</div>
        <p style="margin-bottom:24px;">A full-stack machine learning application for cardiovascular risk screening.</p>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("""
            <div class="card card-red">
                <div class="eyebrow">Mission</div>
                <div class="section-title">What Is PulseIQ?</div>
                <p>PulseIQ uses a trained <strong style="color:#fda4af;">Random Forest classifier</strong> to assess
                cardiovascular disease risk from clinical parameters. Built to raise cardiac awareness and
                assist early-stage screening — while always encouraging professional consultation.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="card">
                <div class="eyebrow">Model Performance</div>
                <div class="section-title">ML Metrics</div>
                <div class="pill-row">
                    <span class="pill pill-green">Accuracy: ~86%</span>
                    <span class="pill pill-blue">Algorithm: Random Forest</span>
                    <span class="pill pill-amber">Dataset: UCI Heart Disease</span>
                </div>
                <p style="margin-top:12px;">
                Trained on the UCI Heart Disease dataset (303 samples, 14 features).
                The model uses <code style="background:rgba(255,255,255,0.05);padding:2px 6px;
                border-radius:4px;color:#93c5fd;">predict_proba()</code> to return a
                continuous risk percentage rather than a binary result, giving far more
                actionable clinical insight.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="card">
                <div class="eyebrow">Tech Stack</div>
                <div class="section-title">Built With</div>
                <div class="pill-row">
                    <span class="pill pill-green">🐍 Python</span>
                    <span class="pill pill-blue">⚡ Streamlit</span>
                    <span class="pill pill-red">🤖 Scikit-learn</span>
                    <span class="pill pill-green">🌲 Random Forest</span>
                    <span class="pill pill-blue">📊 Plotly</span>
                    <span class="pill pill-amber">🗄️ SQLite</span>
                    <span class="pill pill-green">🐼 Pandas</span>
                    <span class="pill pill-blue">📄 FPDF2</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="card">
                <div class="eyebrow">Features</div>
                <div class="section-title">Key Capabilities</div>
                <div class="feat-grid">
                    <div class="feat-card"><div class="feat-icon">🔐</div><div class="feat-label">Secure Auth</div></div>
                    <div class="feat-card"><div class="feat-icon">🪪</div><div class="feat-label">Patient IDs</div></div>
                    <div class="feat-card"><div class="feat-icon">🫀</div><div class="feat-label">AI Prediction</div></div>
                    <div class="feat-card"><div class="feat-icon">📊</div><div class="feat-label">Risk Gauge</div></div>
                    <div class="feat-card"><div class="feat-icon">🗄️</div><div class="feat-label">SQLite DB</div></div>
                    <div class="feat-card"><div class="feat-icon">📋</div><div class="feat-label">History</div></div>
                    <div class="feat-card"><div class="feat-icon">🩺</div><div class="feat-label">Doctor View</div></div>
                    <div class="feat-card"><div class="feat-icon">📄</div><div class="feat-label">PDF Export</div></div>
                    <div class="feat-card"><div class="feat-icon">⚖️</div><div class="feat-label">BMI Calc</div></div>
                    <div class="feat-card"><div class="feat-icon">🌙</div><div class="feat-label">Dark UI</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    #  LOGOUT
    # ══════════════════════════════════════════
    elif "Logout" in menu:

        st.markdown("""
        <div class="page-title">Sign Out</div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card" style="max-width:460px;">
            <div class="eyebrow">Active Session</div>
            <div class="section-title">Confirm Sign Out</div>
            <p>You are signed in as <strong style="color:#fda4af;">{user['full_name']}</strong>
            ({user['patient_id']}). Are you sure you want to end your session?</p>
        </div>
        """, unsafe_allow_html=True)

        cc1, cc2, _ = st.columns([1,1,2])
        with cc1:
            if st.button("Yes, Sign Out"):
                st.session_state.logged_in = False
                st.session_state.user = None
                st.session_state.show_result = False
                st.session_state.last_prediction = None
                st.rerun()
        with cc2:
            if st.button("Cancel"):
                st.rerun()