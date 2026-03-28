# analytics_engine.py - Edupredict AI
# Transforms raw academic data into insights: averages, attendance %, subject performance,
# section/branch-year statistics.

import pandas as pd
import numpy as np

# Pass marks by exam type: MID 14/40, SEM 21/60
PASS_MARKS = {"mid": (14, 40), "sem": (21, 60)}

# Attendance must be at least 75% or it is considered a problem (risk, eligibility, etc.)
MIN_ATTENDANCE_PCT = 75


def get_pass_threshold(exam_name):
    """Return (pass_marks, max_marks) for exam_name. exam_name is case-insensitive (MID/mid, SEM/sem)."""
    if exam_name is None or (isinstance(exam_name, float) and np.isnan(exam_name)):
        return None, None
    key = str(exam_name).strip().lower()
    return PASS_MARKS.get(key, (None, None))


def records_to_dataframe(records):
    """List of academic_records dicts -> DataFrame."""
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def student_level_analytics(df):
    """Per-student: averages, attendance %, subject performance."""
    if df.empty or "student_id" not in df.columns:
        return pd.DataFrame()
    out = []
    for sid, g in df.groupby("student_id"):
        row = {
            "student_id": sid,
            "student_name": g["student_name"].iloc[0] if "student_name" in g.columns else "",
            "year": g["year"].iloc[0] if "year" in g.columns else None,
            "branch": g["branch"].iloc[0] if "branch" in g.columns else None,
            "section": g["section"].iloc[0] if "section" in g.columns else None,
        }
        if "marks" in g.columns:
            row["average_marks"] = g["marks"].mean()
            row["total_subjects"] = g["marks"].count()
        if "attendance" in g.columns and g["attendance"].notna().any():
            row["attendance_pct"] = g["attendance"].mean()
        else:
            row["attendance_pct"] = None
        out.append(row)
    return pd.DataFrame(out)


def subject_performance(df):
    """Subject-wise: average marks per subject."""
    if df.empty or "subject" not in df.columns or "marks" not in df.columns:
        return pd.DataFrame()
    return df.groupby("subject")["marks"].agg(["mean", "count"]).reset_index().rename(columns={"mean": "average_marks", "count": "count"})


def section_level_analytics(df):
    """Section/class: section average, distribution."""
    if df.empty:
        return pd.DataFrame()
    student_avg = student_level_analytics(df)
    if student_avg.empty:
        return pd.DataFrame()
    return student_avg.groupby(["year", "branch", "section"]).agg(
        section_avg_marks=("average_marks", "mean"),
        section_avg_attendance=("attendance_pct", "mean"),
        student_count=("student_id", "count"),
    ).reset_index()


def branch_year_analytics(df):
    """Branch-year: branch performance trend."""
    if df.empty:
        return pd.DataFrame()
    student_avg = student_level_analytics(df)
    if student_avg.empty:
        return pd.DataFrame()
    return student_avg.groupby(["year", "branch"]).agg(
        branch_avg_marks=("average_marks", "mean"),
        branch_avg_attendance=("attendance_pct", "mean"),
        student_count=("student_id", "count"),
    ).reset_index()


def section_vs_branch_totals(df):
    """Section vs branch: for each (branch, section) return total/average marks (for section vs branch chart)."""
    if df.empty or "marks" not in df.columns:
        return pd.DataFrame()
    if "branch" not in df.columns or "section" not in df.columns:
        return pd.DataFrame()
    return df.groupby(["branch", "section"]).agg(
        total_marks=("marks", "sum"),
        average_marks=("marks", "mean"),
        record_count=("marks", "count"),
    ).reset_index()


def exam_trend_by_type(df, exam_type):
    """Filter by exam_name (mid/sem) and return trend: exam_date vs marks, per student or class.
    exam_type: 'mid' or 'sem' (case-insensitive).
    """
    if df.empty or "marks" not in df.columns or "exam_name" not in df.columns:
        return pd.DataFrame()
    key = str(exam_type).strip().lower()
    mask = df["exam_name"].astype(str).str.strip().str.lower() == key
    sub = df.loc[mask].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["marks"] = pd.to_numeric(sub["marks"], errors="coerce")
    sub = sub.dropna(subset=["marks"])
    if "exam_date" in sub.columns and sub["exam_date"].notna().any():
        sub["exam_date_parsed"] = pd.to_datetime(sub["exam_date"], errors="coerce")
        sub = sub.dropna(subset=["exam_date_parsed"])
    return sub


def risk_from_marks_attendance(avg_marks, attendance_pct, exam_name=None):
    """
    Derive risk level from average marks and attendance.
    Attendance must be at least MIN_ATTENDANCE_PCT (75%) or it is considered a problem.
    When exam_name is MID/SEM, compares against pass marks (MID 14/40, SEM 21/60).
    Returns: 'Low Risk', 'Medium Risk', 'High Risk'
    """
    if attendance_pct is None:
        attendance_pct = 80
    if avg_marks is None or np.isnan(avg_marks):
        avg_marks = 50
    pass_marks, max_marks = get_pass_threshold(exam_name)
    if pass_marks is not None and max_marks is not None:
        pass_pct = (pass_marks / max_marks) * 100
        avg_pct = (avg_marks / max_marks) * 100 if max_marks else avg_marks
        low_pct = min(75, pass_pct + 25)
    else:
        avg_pct, pass_pct, low_pct = avg_marks, 40, 60
    # Attendance below minimum (75%) is always a problem: at least Medium Risk, High if marks also low
    if attendance_pct < MIN_ATTENDANCE_PCT:
        if avg_pct < (pass_pct if pass_marks is not None else 40):
            return "High Risk"
        return "Medium Risk"
    if avg_pct < (pass_pct if pass_marks is not None else 40):
        return "High Risk"
    if avg_pct >= low_pct:
        return "Low Risk"
    return "Medium Risk"


def strength_weak_subjects(df, student_id=None):
    """For a student or overall: strongest and weakest subjects by average."""
    if df.empty or "subject" not in df.columns or "marks" not in df.columns:
        return [], []
    sub = df[df["student_id"] == student_id] if student_id else df
    if sub.empty:
        return [], []
    by_sub = sub.groupby("subject")["marks"].mean().sort_values(ascending=False)
    if len(by_sub) == 0:
        return [], []
    strong = by_sub.head(3).index.tolist()
    weak = by_sub.tail(3).index.tolist()
    return strong, weak


def improvement_suggestions_student(avg_marks, attendance_pct, weak_subjects):
    """Student-level improvement suggestions. Attendance must be at least 75% or it is a problem."""
    suggestions = []
    if avg_marks and avg_marks < 50:
        suggestions.append("Focus on consistent study schedule and revision of core topics.")
    if attendance_pct is not None and attendance_pct < 75:
        suggestions.append("Attendance must be at least 75% — below this is a serious problem. Improve attendance immediately.")
    if weak_subjects:
        suggestions.append(f"Prioritize improvement in: {', '.join(weak_subjects)}.")
    if not suggestions:
        suggestions.append("Keep up the good performance and maintain consistency.")
    return suggestions


def improvement_suggestions_section(at_risk_students, common_weak_subjects):
    """Section-level: at-risk list and common weak subjects."""
    suggestions = []
    if at_risk_students:
        suggestions.append(f"At-risk students (consider intervention): {', '.join(at_risk_students[:10])}{'...' if len(at_risk_students) > 10 else ''}")
    if common_weak_subjects:
        suggestions.append(f"Common weak subjects in section: {', '.join(common_weak_subjects)}.")
    return suggestions
