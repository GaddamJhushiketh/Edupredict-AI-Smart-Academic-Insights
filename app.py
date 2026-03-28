# app.py - Edupredict AI: Smart Academic Insights
# Web-based academic analytics and prediction for academies and institutions.
# Roles: Principal (all branches), HOD (assigned branch). No student/teacher access.

import streamlit as st
import pandas as pd
import json
import io
import base64
from datetime import datetime

from database import (
    init_db,
    validate_user,
    add_user,
    get_principal_email,
    delete_principal_users,
    create_hod_request,
    get_pending_hod_requests,
    approve_hod_request,
    reject_hod_request,
    get_students_filtered,
    get_student_by_id,
    get_academic_records,
    get_distinct_years,
    get_distinct_branches,
    get_distinct_sections,
    insert_student,
    insert_academic_record,
    insert_academic_record_with_batch,
    create_upload_batch,
    finalize_upload_batch,
    list_upload_batches,
    delete_upload_batch,
    clear_dataset,
    save_prediction_result,
    get_predictions_for_students,
)
from analytics_engine import (
    records_to_dataframe,
    student_level_analytics,
    subject_performance,
    section_level_analytics,
    branch_year_analytics,
    section_vs_branch_totals,
    exam_trend_by_type,
    get_pass_threshold,
    MIN_ATTENDANCE_PCT,
    risk_from_marks_attendance,
    strength_weak_subjects,
    improvement_suggestions_student,
    improvement_suggestions_section,
)
from werkzeug.security import generate_password_hash

# Optional ML model (legacy pass/fail)
try:
    import joblib
    model = joblib.load("model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    HAS_ML_MODEL = True
except Exception:
    model = None
    model_columns = None
    HAS_ML_MODEL = False

# ---------- Page config & theme ----------
st.set_page_config(
    page_title="Edupredict AI – Smart Academic Insights",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern, clean UI: typography + cards + tasteful gradients
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    :root{
        --bg0:#f7f9ff;
        --bg1:#ffffff;
        --text:#0f172a;
        --muted:#475569;
        --primary:#2563eb;
        --primary2:#1d4ed8;
        --ring:rgba(37,99,235,.25);
        --border:rgba(15,23,42,.10);
        --shadow: 0 10px 30px rgba(2, 6, 23, .08);
        --shadow2: 0 6px 18px rgba(2, 6, 23, .08);
        --radius: 16px;
    }

    html, body, [class*="css"]  { font-family: "Inter", system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
    .main { background-color: var(--bg0); }
    .stApp { background: radial-gradient(1200px 800px at 20% -10%, rgba(37,99,235,.14), transparent 55%),
                     radial-gradient(900px 600px at 110% 10%, rgba(56,189,248,.12), transparent 40%),
                     linear-gradient(180deg, #f8fbff 0%, #ffffff 40%, #f8fafc 100%); }

    /* Layout spacing */
    .block-container { padding: 1.3rem 2.1rem; max-width: 1400px; }

    /* Typography */
    h1, h2, h3 { color: var(--text); letter-spacing: -0.02em; }
    h1 { font-weight: 800; }
    h2 { font-weight: 750; }
    h3 { font-weight: 700; }
    p, li, label, div { color: var(--text); }
    small, .stCaption, .stMarkdown span { color: var(--muted); }

    /* Hide Streamlit chrome */
    
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header [data-testid="stToolbar"] { display: none !important; }
    /* Sidebar */
    div[data-testid="stSidebar"]{
        background: linear-gradient(180deg, rgba(37,99,235,.10) 0%, rgba(255,255,255,.90) 38%, #ffffff 100%);
        border-right: 1px solid var(--border);
    }
    div[data-testid="stSidebar"] .block-container{
        padding-top: 1.2rem;
    }

    /* Inputs */
    div[data-baseweb="select"] > div,
    .stTextInput input,
    .stNumberInput input,
    .stDateInput input,
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
        background: rgba(255,255,255,.9) !important;
        box-shadow: none !important;
    }
    .stTextInput input:focus,
    .stTextArea textarea:focus {
        outline: none !important;
        border-color: rgba(37,99,235,.45) !important;
        box-shadow: 0 0 0 4px var(--ring) !important;
    }

    /* Buttons */
    .stButton > button{
        background: linear-gradient(180deg, var(--primary) 0%, var(--primary2) 100%) !important;
        color: #fff !important;
        border: 1px solid rgba(255,255,255,.15) !important;
        border-radius: 12px !important;
        padding: 0.55rem 0.9rem !important;
        font-weight: 650 !important;
        box-shadow: 0 8px 18px rgba(37,99,235,.25) !important;
        transition: transform .06s ease, filter .12s ease;
    }
    .stButton > button:hover{
        filter: brightness(1.04);
        transform: translateY(-1px);
    }
    .stButton > button:active{ transform: translateY(0px); }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"]{
        gap: 6px;
        background: rgba(255,255,255,.55);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 6px;
        box-shadow: var(--shadow2);
        backdrop-filter: blur(10px);
    }
    .stTabs [data-baseweb="tab"]{
        border-radius: 999px;
        padding: 10px 14px;
        color: var(--muted);
        font-weight: 650;
    }
    .stTabs [aria-selected="true"]{
        background: rgba(37,99,235,.10) !important;
        color: var(--text) !important;
        box-shadow: inset 0 0 0 1px rgba(37,99,235,.18);
    }

    /* Expanders */
    details{
        border-radius: var(--radius) !important;
        border: 1px solid var(--border) !important;
        background: rgba(255,255,255,.86) !important;
        box-shadow: var(--shadow2);
        overflow: hidden;
    }
    details > summary{
        padding: 0.75rem 0.9rem !important;
    }

    /* DataFrame */
    .stDataFrame{
        border-radius: var(--radius);
        border: 1px solid var(--border);
        overflow: hidden;
        box-shadow: var(--shadow2);
        background: rgba(255,255,255,.92);
    }

    /* Custom semantic styles used in app */
    .metric-card {
        background: rgba(255,255,255,.92);
        padding: 1rem 1.1rem;
        border-radius: var(--radius);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
    }
    .auth-wrap{
        max-width: 980px;
        margin: 0 auto;
    }
    .auth-hero{
        padding: 0.3rem 0 1rem 0;
    }
    .auth-card{
        background: rgba(255,255,255,.92);
        border: 1px solid var(--border);
        border-radius: 22px;
        box-shadow: var(--shadow);
        padding: 1.2rem 1.2rem;
        backdrop-filter: blur(10px);
    }
    .auth-kicker{
        display:inline-block;
        font-size: 0.85rem;
        font-weight: 700;
        color: #0b2a69;
        background: rgba(37,99,235,.10);
        border: 1px solid rgba(37,99,235,.18);
        padding: 0.28rem 0.55rem;
        border-radius: 999px;
        margin-bottom: 0.65rem;
    }
    .auth-subtle{
        color: var(--muted);
        font-size: 0.96rem;
        margin-top: -0.25rem;
    }
    .risk-low { color: #0f766e; font-weight: 750; }
    .risk-medium { color: #b45309; font-weight: 750; }
    .risk-high { color: #b91c1c; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# ---------- Init DB ----------
init_db()

# ---------- Session state ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None  # {name, email, role, assigned_branch}
if "selection" not in st.session_state:
    st.session_state.selection = {"year": None, "branch": None, "section": None, "student_id": None}

# ---------- Helpers: branch list for current user ----------
def get_branch_list():
    if not st.session_state.user:
        return None
    if st.session_state.user["role"] == "Principal":
        return None  # all branches
    b = st.session_state.user.get("assigned_branch")
    return [b] if b else []

def get_records_for_selection():
    branch_list = get_branch_list()
    year = st.session_state.selection.get("year")
    branch = st.session_state.selection.get("branch")
    section = st.session_state.selection.get("section")
    student_id = st.session_state.selection.get("student_id")
    records = get_academic_records(year=year, branch=branch, section=section, branch_list=branch_list)
    if student_id:
        records = [r for r in records if r["student_id"] == student_id]
    return records

def get_students_for_selection():
    branch_list = get_branch_list()
    return get_students_filtered(
        year=st.session_state.selection.get("year"),
        branch=st.session_state.selection.get("branch"),
        section=st.session_state.selection.get("section"),
        branch_list=branch_list,
    )

# ---------- Auth: Login / Register / HOD Request ----------
if not st.session_state.logged_in:
    st.markdown("<div class='auth-wrap'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='auth-hero'>
      <h1 style="margin:0;">Edupredict AI - Smart Academic Insights</h1>
      <div class='auth-subtle'>Sign in to view analytics, upload data, and generate reports. HOD accounts require Principal approval.</div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
        tab_login, tab_principal, tab_hod = st.tabs(["Login", "Register Principal", "Request HOD"])

        with tab_login:
            st.markdown("### Welcome back")
            st.caption("Use your approved Principal/HOD account.")
            with st.form("login_form", clear_on_submit=False):
                email = st.text_input("Email", placeholder="you@example.com")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("Login")
            if submitted:
                user = validate_user(email, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user = user
                    st.success(f"Welcome, {user['name']} ({user['role']})")
                    st.rerun()
                else:
                    st.error("Invalid credentials or account not approved.")

        with tab_principal:
            st.markdown("### Create Principal account")
            st.caption("Only one Principal account exists at a time.")
            if get_principal_email():
                st.info("A Principal account already exists.")
                st.markdown("If you want to replace it, remove the current Principal and register yours.")
                with st.form("replace_principal_form", clear_on_submit=False):
                    replace = st.form_submit_button("Replace Principal (enable registration)")
                if replace:
                    delete_principal_users()
                    st.success("Existing Principal removed. Register below.")
                    st.rerun()
            else:
                with st.form("principal_register_form", clear_on_submit=False):
                    name = st.text_input("Full name", placeholder="Your name")
                    email = st.text_input("Email", placeholder="yourname@gmail.com")
                    password = st.text_input("Password", type="password", placeholder="Create a strong password")
                    submitted = st.form_submit_button("Create account")
                if submitted:
                    if not all([name, email, password]):
                        st.error("Fill all fields.")
                    else:
                        add_user(
                            name=name.strip(),
                            email=email.strip(),
                            password=password,
                            role="Principal",
                            assigned_branch=None,
                            approval_status="approved",
                        )
                        st.success("Principal account created! Please login.")
                        st.rerun()

        with tab_hod:
            st.markdown("### Request HOD access")
            principal_email = get_principal_email() or ""
            st.caption("Your request will appear for the Principal to approve.")
            with st.form("hod_request_form", clear_on_submit=False):
                name = st.text_input("Full name", placeholder="Your name")
                email = st.text_input("Email", placeholder="you@example.com")
                branch = st.text_input("Branch", placeholder="e.g. CSE, ECE")
                password = st.text_input("Password", type="password", placeholder="Create a password for your account")
                principal = st.text_input("Principal email (for approval)", value=principal_email, placeholder="principal@example.com")
                submitted = st.form_submit_button("Submit request")
            if submitted:
                if not all([name, email, branch, password]):
                    st.error("Fill all fields.")
                elif not principal:
                    st.error("Enter Principal email for approval.")
                else:
                    pw_hash = generate_password_hash(password)
                    create_hod_request(email.strip(), name.strip(), branch.strip(), principal.strip(), pw_hash)
                    st.success("Request sent. Wait for Principal approval.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
        st.markdown("### What you can do")
        st.markdown("""
        - **Upload academic data** (CSV/Excel) in supported formats  
        - **Student insights**: subject performance, attendance vs marks  
        - **Class analytics**: trends, comparisons, distributions  
        - **Manage uploads**: delete an import batch when needed  
        """)
        st.markdown("---")
        st.markdown("### Roles")
        st.markdown("""
        - **Principal**: full access across branches + approve HOD requests  
        - **HOD**: access limited to your assigned branch  
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ---------- Sidebar: user & navigation ----------
user = st.session_state.get("user")
if not user:
    st.warning("Session expired. Please log in again.")
    st.session_state.logged_in = False
    st.rerun()

branch_list = get_branch_list()
st.sidebar.markdown("## 🎓 Edupredict AI")
st.sidebar.caption("Smart Academic Insights")
st.sidebar.markdown(f"**{user['name']}** ({user['role']})")
if user.get("assigned_branch"):
    st.sidebar.caption(f"Branch: {user['assigned_branch']}")
st.sidebar.markdown("---")

nav_options = ["📊 Analytics Dashboard", "📤 Upload Data", "🗑️ Manage Uploads"]
if user["role"] == "Principal":
    nav_options.append("👑 Approve HOD (Principal)")
    nav_options.append("🧹 Clear Dataset (Principal)")
nav_options.append("🚪 Logout")
nav = st.sidebar.radio("Navigation", nav_options)

if nav == "🚪 Logout":
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()

# ---------- Principal: Approve HOD ----------
if nav == "👑 Approve HOD (Principal)":
    if user["role"] != "Principal":
        st.warning("Only Principal can approve HOD requests.")
        st.stop()
    st.header("Approve HOD account requests")
    principal_email = user["email"]
    requests_list = get_pending_hod_requests(principal_email)
    if not requests_list:
        st.info("No pending HOD requests.")
        st.stop()
    for req in requests_list:
        with st.expander(f"Request: {req['requested_by_email']} – Branch: {req['branch']}"):
            st.write(f"Name: {req['requested_by_name']} | Branch: {req['branch']}")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Approve", key=f"approve_{req['id']}"):
                    ok = approve_hod_request(req["id"])
                    if ok:
                        st.success("HOD account approved.")
                        st.rerun()
                    else:
                        st.error("Could not approve this request. The email may already be registered.")
            with c2:
                if st.button("Reject", key=f"reject_{req['id']}"):
                    reject_hod_request(req["id"])
                    st.info("Request rejected.")
                    st.rerun()
    st.stop()

# ---------- Principal: Clear Dataset ----------
if nav == "🧹 Clear Dataset (Principal)":
    if user["role"] != "Principal":
        st.warning("Only Principal can clear the dataset.")
        st.stop()
    st.header("🧹 Clear previous dataset")
    st.caption("This removes all uploaded academic data (students, records, predictions, upload history). User accounts are kept.")

    st.markdown("**This action cannot be undone.**")
    confirm = st.checkbox("I understand this will permanently delete the dataset", key="confirm_clear_dataset")
    typed = st.text_input("Type `CLEAR` to confirm", key="type_clear_dataset")
    if st.button("Clear dataset now", type="primary", disabled=not (confirm and typed.strip().upper() == "CLEAR")):
        clear_dataset(keep_users=True, keep_hod_requests=True)
        st.success("Dataset cleared. Upload fresh data to begin again.")
        st.rerun()
    st.stop()

# ---------- Upload Data ----------
if nav == "📤 Upload Data":
    st.header("📤 Upload academic data")
    st.caption(
        "Upload one or more CSV/Excel files. Each dataset should have: Name, Year, Branch, Section, Total marks, Attendance, Exam name (MID/SEM), Exam date. "
        "**MID** is every **2 months**, **SEM** every **6 months**. "
        "**Attendance must be at least 75%** — below that is treated as a problem (risk/eligibility). Rows with attendance < 75% will be excluded from save. "
        "Formats: (1) Totals: name, year, branch, section, totalmarks, attendance, exam_name, exam_date. "
        "(2) Subject-wise: name, year, branch, section, subject, marks, attendance (optional), exam_name (optional), exam_date (optional)."
    )
    if user["role"] == "HOD":
        st.info(f"Restricted upload: you can upload **only** branch **{user.get('assigned_branch')}**.")
    uploaded_list = st.file_uploader("Choose file(s)", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
    if uploaded_list:
        import re
        def _norm_col(c: str) -> str:
            c = str(c).strip().lower()
            c = re.sub(r"\s+", "_", c)
            c = re.sub(r"[^a-z0-9_]+", "", c)
            return c

        required = ["name", "year", "branch", "section"]
        processed = []  # list of (filename, df ready to save, format_type)

        for uploaded in uploaded_list:
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)
            except Exception as e:
                st.error(f"Could not read **{getattr(uploaded, 'name', 'file')}**: {e}")
                continue
            df.columns = [_norm_col(c) for c in df.columns]
            if "attendence" in df.columns and "attendance" not in df.columns:
                df = df.rename(columns={"attendence": "attendance"})

            # Totals-only: name, year, branch, section, totalmarks, attendance, exam_name, exam_date
            if "totalmarks" in df.columns and ("subject" not in df.columns and "marks" not in df.columns):
                df = df.copy()
                df["subject"] = "TOTAL"
                df["marks"] = df["totalmarks"]
                df["exam_name"] = df.get("exam_name", "Overall")
                df["exam_date"] = df.get("exam_date", "")

            if user["role"] == "HOD":
                allowed_branch = (user.get("assigned_branch") or "").strip()
                if not allowed_branch:
                    st.error("Your HOD account has no assigned branch. Contact the Principal.")
                    st.stop()
                if "branch" not in df.columns:
                    df = df.copy()
                    df["branch"] = allowed_branch
                before = len(df)
                df["branch"] = df["branch"].astype(str).str.strip()
                df = df[df["branch"].str.casefold() == allowed_branch.casefold()]
                if before - len(df) > 0:
                    st.warning(f"**{uploaded.name}**: Removed {before - len(df)} row(s) not in branch {allowed_branch}.")
                if df.empty:
                    st.warning(f"**{uploaded.name}**: No rows left after branch restriction. Skipped.")
                    continue

            if "subject" in df.columns and "marks" in df.columns:
                missing = [r for r in required if r not in df.columns]
                if missing:
                    st.error(f"**{uploaded.name}**: Missing columns: {missing}. Skipped.")
                    continue
                df["attendance"] = pd.to_numeric(df.get("attendance", float("nan")), errors="coerce")
                df["exam_name"] = df.get("exam_name", "")
                df["exam_date"] = df.get("exam_date", "")
                df = df.dropna(subset=["name", "year", "branch", "section", "subject", "marks"])
                # Enforce attendance >= 75%; exclude rows with attendance present and below threshold
                if "attendance" in df.columns:
                    has_att = df["attendance"].notna()
                    low_att = has_att & (df["attendance"] < MIN_ATTENDANCE_PCT)
                    if low_att.any():
                        n_low = low_att.sum()
                        df = df[~low_att].copy()
                        st.warning(f"**{uploaded.name}**: {n_low} row(s) had attendance < {MIN_ATTENDANCE_PCT}% — excluded (attendance must be ≥ {MIN_ATTENDANCE_PCT}%).")
                if not df.empty:
                    processed.append((uploaded, df, "long"))
            else:
                score_cols = [c for c in df.columns if c not in required and df[c].dtype in ["int64", "float64"]]
                if not score_cols or any(r not in df.columns for r in required):
                    st.error(f"**{uploaded.name}**: Need (name, year, branch, section, subject, marks) or subject columns. Skipped.")
                    continue
                processed.append((uploaded, df, "wide"))

        if processed:
            st.info(f"**Rule:** Attendance must be at least **{MIN_ATTENDANCE_PCT}%**. Rows below that are excluded when saving.")
            for up_name, pdf, fmt in processed:
                with st.expander(f"Preview: **{up_name.name}** ({len(pdf)} rows)", expanded=(len(processed) == 1)):
                    st.dataframe(pdf.head(20), use_container_width=True)

            if st.button("Save all to database"):
                total_r = 0
                for uploaded, pdf, fmt in processed:
                    branches_in_upload = sorted({str(b).strip() for b in pdf["branch"].dropna().unique().tolist()})
                    years_in_upload = sorted({str(y).strip() for y in pdf["year"].dropna().unique().tolist()})
                    batch_id = create_upload_batch(
                        uploaded_by_user_id=user.get("user_id"),
                        filename=getattr(uploaded, "name", "uploaded_file"),
                        branches=branches_in_upload,
                        years=years_in_upload,
                    )
                    count_r = 0
                    if fmt == "long":
                        for _, row in pdf.iterrows():
                            sid = insert_student(
                                str(row["name"]), str(row["year"]), str(row["branch"]), str(row["section"]))
                            if sid:
                                insert_academic_record_with_batch(
                                    sid,
                                    str(row["subject"]),
                                    float(row["marks"]),
                                    float(row["attendance"]) if pd.notna(row.get("attendance")) else None,
                                    str(row.get("exam_name", "")) or None,
                                    str(row.get("exam_date", "")) or None,
                                    upload_batch_id=batch_id,
                                )
                                count_r += 1
                    else:
                        score_cols = [c for c in pdf.columns if c not in required and pdf[c].dtype in ["int64", "float64"]]
                        for _, row in pdf.iterrows():
                            sid = insert_student(
                                str(row["name"]), str(row["year"]), str(row["branch"]), str(row["section"]))
                            if sid:
                                for sub in score_cols:
                                    val = row.get(sub)
                                    if pd.notna(val):
                                        insert_academic_record_with_batch(sid, sub, float(val), None, None, None, upload_batch_id=batch_id)
                                        count_r += 1
                    finalize_upload_batch(batch_id, rows_inserted=count_r)
                    total_r += count_r
                st.success(f"Saved {len(processed)} dataset(s); total {total_r} academic records inserted.")
                st.rerun()
        else:
            st.warning("No valid datasets to save. Check column names and branch filters.")
    st.stop()

# ---------- Manage Uploads ----------
if nav == "🗑️ Manage Uploads":
    st.header("🗑️ Manage uploaded files (delete imported data)")
    st.caption("Uploads are stored as database import batches. Deleting here removes the imported rows from the database (the file itself is not stored).")
    branch_list = get_branch_list()
    batches = list_upload_batches(branch_list=branch_list, limit=100)
    if not batches:
        st.info("No upload history found yet.")
        st.stop()

    st.write("Select an upload batch to delete.")
    for b in batches:
        title = f"#{b['id']} – {b.get('filename')} – {b.get('created_at')}"
        meta = f"Branches: {b.get('branches') or '—'} | Years: {b.get('years') or '—'} | Rows inserted: {b.get('rows_inserted')}"
        with st.expander(title):
            st.caption(meta)
            confirm = st.checkbox("I understand this will delete imported data", key=f"confirm_del_{b['id']}")
            if st.button("Delete this upload", type="primary", disabled=not confirm, key=f"del_{b['id']}"):
                res = delete_upload_batch(b["id"], branch_list=branch_list)
                if not res.get("ok"):
                    st.error(res.get("error") or "Delete failed.")
                else:
                    st.success(
                        f"Deleted {res.get('deleted_records', 0)} record(s), "
                        f"{res.get('deleted_students', 0)} orphan student(s), "
                        f"{res.get('deleted_predictions', 0)} prediction(s)."
                    )
                    st.rerun()
    st.stop()

# ---------- Analytics Dashboard: Year → Branch → Section → Student ----------
st.header("📊 Analytics Dashboard")
st.caption("Select Year → Branch → Section → Student (optional) to view analytics.")

years = get_distinct_years(branch_list)
branches = get_distinct_branches(branch_list)

col1, col2, col3, col4 = st.columns(4)
with col1:
    year = st.selectbox("Year", ["-- Select --"] + (years or []), key="sel_year")
with col2:
    if user["role"] == "HOD":
        fixed_branch = (user.get("assigned_branch") or "").strip()
        branch = st.selectbox("Branch", [fixed_branch] if fixed_branch else ["-- No branch assigned --"], disabled=True, key="sel_branch")
    else:
        branch = st.selectbox("Branch", ["-- Select --"] + (branches or []), key="sel_branch")
with col3:
    sections = get_distinct_sections(
        year=year if year != "-- Select --" else None,
        branch=branch if branch != "-- Select --" else None,
        branch_list=branch_list,
    )
    section = st.selectbox("Section", ["-- Select --"] + (sections or []), key="sel_section")

# Update selection so get_students_for_selection and get_records_for_selection use current filters
st.session_state.selection["year"] = year if year != "-- Select --" else None
st.session_state.selection["branch"] = branch if branch != "-- Select --" else None
st.session_state.selection["section"] = section if section != "-- Select --" else None
st.session_state.selection["student_id"] = None

with col4:
    students = get_students_for_selection() if (st.session_state.selection["year"] and st.session_state.selection["branch"] and st.session_state.selection["section"]) else []
    student_options = ["-- All --"] + [f"{s['name']} (ID: {s['student_id']})" for s in students]
    student_choice = st.selectbox("Student", student_options, key="sel_student")

if student_choice != "-- All --" and students:
    for s in students:
        if f"{s['name']} (ID: {s['student_id']})" == student_choice:
            st.session_state.selection["student_id"] = s["student_id"]
            break

records = get_records_for_selection()
df_rec = records_to_dataframe(records)

if df_rec.empty:
    st.info("No data for the selected filters. Upload data or change selection.")
    st.stop()

# ---------- Six tabs ----------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dataset View",
    "📈 Graph Analytics",
    "🤖 AI Prediction Report",
    "📊 Performance Report",
    "💡 Improvement Suggestions",
    "📥 Download & Export",
])

with tab1:
    st.subheader("Dataset View")
    st.dataframe(df_rec, use_container_width=True)
    search = st.text_input("Search in table (filter)")
    if search:
        mask = df_rec.astype(str).apply(lambda x: x.str.contains(search, case=False, na=False)).any(axis=1)
        st.dataframe(df_rec[mask], use_container_width=True)
    csv = df_rec.to_csv(index=False).encode("utf-8")
    st.download_button("Export CSV", csv, "edupredict_dataset.csv", "text/csv")

with tab2:
    st.subheader("Graph Categories")
    import matplotlib.pyplot as plt

    df_plot = df_rec.copy()
    if "marks" in df_plot.columns:
        df_plot["marks"] = pd.to_numeric(df_plot["marks"], errors="coerce")
    if "attendance" in df_plot.columns:
        df_plot["attendance"] = pd.to_numeric(df_plot["attendance"], errors="coerce")

    student_analytics = student_level_analytics(df_plot)
    if student_analytics.empty:
        st.info("Not enough data for charts.")
        st.stop()

    # ---------------- Section vs Branch ----------------
    st.markdown("### Section vs Branch (average marks)")
    sec_branch = section_vs_branch_totals(df_plot)
    if sec_branch.empty:
        st.info("No branch/section data for chart.")
    else:
        sec_branch["branch_section"] = sec_branch["branch"].astype(str) + " – " + sec_branch["section"].astype(str)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = range(len(sec_branch))
        ax.bar(x, sec_branch["average_marks"], color="#34a853", edgecolor="#0d8050")
        ax.set_xticks(x)
        ax.set_xticklabels(sec_branch["branch_section"], rotation=35, ha="right")
        ax.set_ylabel("Average marks")
        ax.set_xlabel("Branch – Section")
        ax.set_title("Section vs Branch — average marks per branch and section")
        ax.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ---------------- Attendance vs marks ----------------
    st.markdown("### Attendance vs marks")
    if "attendance_pct" not in student_analytics.columns or student_analytics["attendance_pct"].isna().all():
        st.info("Attendance values are missing in the dataset.")
    else:
        scat = student_analytics[["student_name", "average_marks", "attendance_pct"]].dropna()
        if scat.empty:
            st.info("Not enough attendance + marks data to plot.")
        else:
            fig, ax = plt.subplots(figsize=(7.5, 4))
            ax.scatter(scat["attendance_pct"], scat["average_marks"], alpha=0.85, s=55, c="#2563eb", edgecolors="#1d4ed8")
            ax.set_xlabel("Attendance (%)")
            ax.set_ylabel("Average marks")
            ax.set_title("Attendance vs marks (student averages)")
            ax.grid(True, alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ---------------- Student MID graph & Student SEM graph ----------------
    st.markdown("### Exam trends (MID & SEM) — improvement / decline over time")
    st.caption("MID: every 2 months | SEM: every 6 months")
    mid_pass, mid_max = get_pass_threshold("mid")   # 14, 40
    sem_pass, sem_max = get_pass_threshold("sem")   # 21, 60

    c_mid, c_sem = st.columns(2)
    with c_mid:
        st.markdown("**Student MID graph**")
        mid_df = exam_trend_by_type(df_plot, "mid")
        if mid_df.empty:
            st.info("No MID exam data (exam_name = MID and exam_date in upload).")
        else:
            fig, ax = plt.subplots(figsize=(7.5, 4))
            by_date = mid_df.groupby("exam_date_parsed")["marks"].mean().reset_index().sort_values("exam_date_parsed")
            ax.plot(by_date["exam_date_parsed"], by_date["marks"], marker="s", linewidth=2.5, color="#1e293b", label="Class average")
            n_students = mid_df["student_id"].nunique()
            for sid, g in mid_df.groupby("student_id"):
                g = g.sort_values("exam_date_parsed")
                label = g["student_name"].iloc[0] if (st.session_state.selection.get("student_id") and n_students == 1) else f"Student {sid}"
                ax.plot(g["exam_date_parsed"], g["marks"], marker="o", alpha=0.85, label=label)
            if mid_pass is not None:
                ax.axhline(mid_pass, color="#b91c1c", linestyle="--", linewidth=1.5, label=f"MID pass ({mid_pass}/{mid_max})")
            ax.set_xlabel("Exam date")
            ax.set_ylabel("Marks")
            ax.set_title("MID exam trend — improvement/decline over time")
            ax.legend(loc="best", fontsize=7)
            ax.grid(True, alpha=0.25)
            plt.xticks(rotation=25)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with c_sem:
        st.markdown("**Student SEM graph**")
        sem_df = exam_trend_by_type(df_plot, "sem")
        if sem_df.empty:
            st.info("No SEM exam data (exam_name = SEM and exam_date in upload).")
        else:
            fig, ax = plt.subplots(figsize=(7.5, 4))
            by_date = sem_df.groupby("exam_date_parsed")["marks"].mean().reset_index().sort_values("exam_date_parsed")
            ax.plot(by_date["exam_date_parsed"], by_date["marks"], marker="s", linewidth=2.5, color="#1e293b", label="Class average")
            n_students = sem_df["student_id"].nunique()
            for sid, g in sem_df.groupby("student_id"):
                g = g.sort_values("exam_date_parsed")
                label = g["student_name"].iloc[0] if (st.session_state.selection.get("student_id") and n_students == 1) else f"Student {sid}"
                ax.plot(g["exam_date_parsed"], g["marks"], marker="o", alpha=0.85, label=label)
            if sem_pass is not None:
                ax.axhline(sem_pass, color="#b91c1c", linestyle="--", linewidth=1.5, label=f"SEM pass ({sem_pass}/{sem_max})")
            ax.set_xlabel("Exam date")
            ax.set_ylabel("Marks")
            ax.set_title("SEM exam trend — improvement/decline over time")
            ax.legend(loc="best", fontsize=7)
            ax.grid(True, alpha=0.25)
            plt.xticks(rotation=25)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ---------------- Class average trend ----------------
    st.markdown("### Class average trend")
    trend_df = df_plot.copy()
    by = None
    if "exam_date" in trend_df.columns and trend_df["exam_date"].notna().any():
        trend_df["exam_date_parsed"] = pd.to_datetime(trend_df["exam_date"], errors="coerce")
        if trend_df["exam_date_parsed"].notna().any():
            by = trend_df.dropna(subset=["exam_date_parsed", "marks"]).groupby("exam_date_parsed")["marks"].mean().reset_index()
            x_label = "Exam date"
    if by is None or by.empty:
        if "exam_name" in trend_df.columns and trend_df["exam_name"].notna().any():
            by2 = trend_df.dropna(subset=["exam_name", "marks"]).groupby("exam_name")["marks"].mean().reset_index()
            if not by2.empty:
                by = by2
                by = by.rename(columns={"exam_name": "exam_date_parsed"})
                x_label = "Exam"
    if by is not None and not by.empty and len(by) >= 2:
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.plot(by["exam_date_parsed"], by["marks"], marker="o", linewidth=2.2, color="#2563eb")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Average marks")
        ax.set_title("Class average trend")
        ax.grid(True, alpha=0.25)
        plt.xticks(rotation=25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No exam timeline (add exam_date or exam_name to upload) for trend.")

    # ---------------- Performance distribution (pass: MID 14/40, SEM 21/60) ----------------
    st.markdown("**Performance distribution**")
    dist = student_analytics[["average_marks"]].dropna()
    if dist.empty:
        st.info("No marks distribution available.")
    else:
        fig, ax = plt.subplots(figsize=(12, 3.8))
        ax.hist(dist["average_marks"], bins=12, color="#38bdf8", edgecolor="#0284c7", alpha=0.9)
        if mid_pass is not None:
            ax.axvline(mid_pass, color="#b91c1c", linestyle="--", linewidth=2, label=f"MID pass ({mid_pass}/{mid_max})")
        if sem_pass is not None:
            ax.axvline(sem_pass, color="#ea580c", linestyle="-.", linewidth=2, label=f"SEM pass ({sem_pass}/{sem_max})")
        ax.set_xlabel("Student average marks")
        ax.set_ylabel("Number of students")
        ax.set_title("Performance distribution (student averages)")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with tab3:
    st.subheader("AI Prediction Report")
    student_analytics = student_level_analytics(df_rec)
    if student_analytics.empty:
        st.info("No student-level aggregates.")
    else:
        for _, row in student_analytics.iterrows():
            avg = row.get("average_marks")
            att = row.get("attendance_pct")
            risk = risk_from_marks_attendance(avg, att)
            pred_score = avg if avg is not None else 0
            summary = f"Average marks: {avg:.1f}" + (f", Attendance: {att:.0f}%" if att else "")
            risk_class = "risk-low" if risk == "Low Risk" else "risk-medium" if risk == "Medium Risk" else "risk-high"
            st.markdown(f"**{row.get('student_name', row['student_id'])}** – Predicted score: **{pred_score:.1f}** – <span class='{risk_class}'>{risk}</span>", unsafe_allow_html=True)
            st.caption(summary)

with tab4:
    st.subheader("Performance Report")
    student_analytics = student_level_analytics(df_rec)
    if not student_analytics.empty:
        ranked = student_analytics.sort_values("average_marks", ascending=False).reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        st.dataframe(ranked[["rank", "student_name", "average_marks", "attendance_pct"]], use_container_width=True)
        for _, row in student_analytics.iterrows():
            strong, weak = strength_weak_subjects(df_rec, row["student_id"])
            st.caption(f"{row.get('student_name')}: Strong – {strong}; Weak – {weak}")
    else:
        st.info("No student-level data.")

with tab5:
    st.subheader("Improvement Suggestions")
    student_analytics = student_level_analytics(df_rec)
    if st.session_state.selection.get("student_id"):
        s = student_analytics[student_analytics["student_id"] == st.session_state.selection["student_id"]]
        if not s.empty:
            r = s.iloc[0]
            _, weak = strength_weak_subjects(df_rec, r["student_id"])
            for sug in improvement_suggestions_student(r.get("average_marks"), r.get("attendance_pct"), weak):
                st.write(f"• {sug}")
        else:
            st.info("No data for selected student.")
    else:
        at_risk = []
        if "average_marks" in student_analytics.columns and "attendance_pct" in student_analytics.columns:
            for _, r in student_analytics.iterrows():
                if risk_from_marks_attendance(r.get("average_marks"), r.get("attendance_pct")) == "High Risk":
                    at_risk.append(r.get("student_name", ""))
        sub_perf = subject_performance(df_rec)
        common_weak = sub_perf[sub_perf["average_marks"] < 50]["subject"].tolist() if not sub_perf.empty else []
        for sug in improvement_suggestions_section(at_risk, common_weak):
            st.write(f"• {sug}")

with tab6:
    st.subheader("Download & Export")
    student_analytics = student_level_analytics(df_rec)
    csv_full = df_rec.to_csv(index=False).encode("utf-8")
    st.download_button("Download dataset (CSV)", csv_full, "edupredict_export.csv", "text/csv")
    if not student_analytics.empty:
        csv_analytics = student_analytics.to_csv(index=False).encode("utf-8")
        st.download_button("Download analytics (CSV, Power BI ready)", csv_analytics, "edupredict_analytics.csv", "text/csv")
    st.caption("PDF report generation can be added with reportlab or similar library.")
