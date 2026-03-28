# database.py - Edupredict AI
# Hierarchical academic data: Year → Branch → Section → Student → Academic Records
# Roles: Principal (all branches), HOD (assigned branch only)

import sqlite3
import json
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = "edupredict_ai.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    # Ensure FK constraints where supported/declared.
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
    except Exception:
        pass
    return conn


def _has_column(conn, table: str, column: str) -> bool:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    return column in cols


def _ensure_column(conn, table: str, column: str, ddl: str):
    """Best-effort migration for simple ALTER TABLE ADD COLUMN use-cases."""
    if _has_column(conn, table, column):
        return
    cur = conn.cursor()
    cur.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")

def init_db():
    conn = get_conn()
    c = conn.cursor()

    # Users: Principal or HOD only (no students/teachers)
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('Principal', 'HOD')),
            assigned_branch TEXT,
            approval_status TEXT DEFAULT 'approved' CHECK(approval_status IN ('approved', 'pending', 'rejected')),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Principal has assigned_branch NULL (sees all). HOD has one branch.

    # HOD account requests (Principal approval workflow)
    c.execute("""
        CREATE TABLE IF NOT EXISTS hod_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            requested_by_email TEXT NOT NULL,
            requested_by_name TEXT,
            branch TEXT NOT NULL,
            principal_email TEXT NOT NULL,
            password_hash TEXT,
            status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'approved', 'rejected')),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Students: academic hierarchy
    c.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            year TEXT NOT NULL,
            branch TEXT NOT NULL,
            section TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name, year, branch, section)
        )
    """)

    # Academic records per student
    c.execute("""
        CREATE TABLE IF NOT EXISTS academic_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            marks REAL NOT NULL,
            attendance REAL,
            exam_name TEXT,
            exam_date TEXT,
            upload_batch_id INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        )
    """)

    # Upload batches: lets us delete/undo an uploaded file import later.
    c.execute("""
        CREATE TABLE IF NOT EXISTS upload_batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uploaded_by_user_id INTEGER,
            filename TEXT NOT NULL,
            branches TEXT,
            years TEXT,
            rows_inserted INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (uploaded_by_user_id) REFERENCES users(user_id)
        )
    """)

    # If DB existed before upload_batch_id was introduced, add it.
    _ensure_column(conn, "academic_records", "upload_batch_id", "upload_batch_id INTEGER")

    # AI prediction results
    c.execute("""
        CREATE TABLE IF NOT EXISTS prediction_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            predicted_score REAL,
            risk_level TEXT CHECK(risk_level IN ('Low Risk', 'Medium Risk', 'High Risk')),
            analysis_summary TEXT,
            input_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        )
    """)

    conn.commit()
    conn.close()


def clear_dataset(keep_users=True, keep_hod_requests=True):
    """Clear previously uploaded academic dataset.

    By default this keeps authentication tables (`users`, `hod_requests`) and removes:
    - students
    - academic_records
    - prediction_results
    - upload_batches
    """
    conn = get_conn()
    c = conn.cursor()

    # Delete in dependency-safe order.
    c.execute("DELETE FROM prediction_results")
    c.execute("DELETE FROM academic_records")
    c.execute("DELETE FROM upload_batches")
    c.execute("DELETE FROM students")

    if not keep_hod_requests:
        c.execute("DELETE FROM hod_requests")
    if not keep_users:
        c.execute("DELETE FROM users")

    conn.commit()
    conn.close()


# ---------- User / Auth ----------
def add_user(name, email, password, role="HOD", assigned_branch=None, approval_status="approved"):
    conn = get_conn()
    c = conn.cursor()
    password_hash = generate_password_hash(password)
    c.execute("""
        INSERT INTO users (name, email, password_hash, role, assigned_branch, approval_status)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (name, email, password_hash, role, assigned_branch, approval_status))
    conn.commit()
    conn.close()


def validate_user(email, password):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT user_id, name, password_hash, role, assigned_branch, approval_status
        FROM users WHERE email = ?
    """, (email,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    user_id, name, password_hash, role, assigned_branch, approval_status = row
    if approval_status != "approved":
        return None
    if check_password_hash(password_hash, password):
        return {
            "user_id": user_id,
            "name": name,
            "email": email,
            "role": role,
            "assigned_branch": assigned_branch,
        }
    return None


def get_user_by_id(user_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT user_id, name, email, role, assigned_branch FROM users WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {"user_id": row[0], "name": row[1], "email": row[2], "role": row[3], "assigned_branch": row[4]}


def get_all_users():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT user_id, name, email, role, assigned_branch, approval_status FROM users")
    rows = c.fetchall()
    conn.close()
    return [
        {"user_id": r[0], "name": r[1], "email": r[2], "role": r[3], "assigned_branch": r[4], "approval_status": r[5]}
        for r in rows
    ]


def get_principal_email():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT email FROM users WHERE role = 'Principal' LIMIT 1")
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def delete_principal_users():
    """Remove all Principal accounts (for replacing with a new Principal)."""
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE role = 'Principal'")
    conn.commit()
    conn.close()


# ---------- HOD request (approval workflow) ----------
def create_hod_request(requested_by_email, requested_by_name, branch, principal_email, password_hash):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO hod_requests (requested_by_email, requested_by_name, branch, principal_email, password_hash, status)
        VALUES (?, ?, ?, ?, ?, 'pending')
    """, (requested_by_email, requested_by_name, branch, principal_email, password_hash))
    conn.commit()
    rid = c.lastrowid
    conn.close()
    return rid


def get_pending_hod_requests(principal_email):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT id, requested_by_email, requested_by_name, branch, principal_email, created_at
        FROM hod_requests WHERE principal_email = ? AND status = 'pending'
        ORDER BY created_at DESC
    """, (principal_email,))
    rows = c.fetchall()
    conn.close()
    return [
        {"id": r[0], "requested_by_email": r[1], "requested_by_name": r[2], "branch": r[3], "principal_email": r[4], "created_at": r[5]}
        for r in rows
    ]


def get_hod_request_by_id(req_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, requested_by_email, requested_by_name, branch, password_hash, status FROM hod_requests WHERE id = ?", (req_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "requested_by_email": row[1], "requested_by_name": row[2], "branch": row[3], "password_hash": row[4], "status": row[5]}


def approve_hod_request(req_id):
    req = get_hod_request_by_id(req_id)
    if not req or req["status"] != "pending" or not req.get("password_hash"):
        return False
    conn = get_conn()
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO users (name, email, password_hash, role, assigned_branch, approval_status)
            VALUES (?, ?, ?, 'HOD', ?, 'approved')
        """, (req["requested_by_name"] or req["requested_by_email"], req["requested_by_email"], req["password_hash"], req["branch"]))
        c.execute("UPDATE hod_requests SET status = 'approved' WHERE id = ?", (req_id,))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Likely email already exists in `users` (unique constraint).
        conn.rollback()
        return False
    finally:
        conn.close()


def reject_hod_request(req_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE hod_requests SET status = 'rejected' WHERE id = ?", (req_id,))
    conn.commit()
    conn.close()


# ---------- Students ----------
def insert_student(name, year, branch, section):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT OR IGNORE INTO students (name, year, branch, section) VALUES (?, ?, ?, ?)
    """, (name, year, branch, section))
    c.execute("SELECT student_id FROM students WHERE name = ? AND year = ? AND branch = ? AND section = ?",
              (name, year, branch, section))
    row = c.fetchone()
    conn.commit()
    sid = row[0] if row else None
    conn.close()
    return sid


def get_students_filtered(year=None, branch=None, section=None, branch_list=None):
    """branch_list: for HOD, list of allowed branches (single). For Principal, None = all."""
    if branch_list is not None and len(branch_list) == 0:
        return []
    conn = get_conn()
    c = conn.cursor()
    q = "SELECT student_id, name, year, branch, section FROM students WHERE 1=1"
    params = []
    if year:
        q += " AND year = ?"
        params.append(year)
    if branch:
        q += " AND branch = ?"
        params.append(branch)
    if section:
        q += " AND section = ?"
        params.append(section)
    if branch_list is not None:
        placeholders = ",".join("?" * len(branch_list))
        q += f" AND branch IN ({placeholders})"
        params.extend(branch_list)
    q += " ORDER BY name"
    c.execute(q, params)
    rows = c.fetchall()
    conn.close()
    return [{"student_id": r[0], "name": r[1], "year": r[2], "branch": r[3], "section": r[4]} for r in rows]


def get_student_by_id(student_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT student_id, name, year, branch, section FROM students WHERE student_id = ?", (student_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {"student_id": row[0], "name": row[1], "year": row[2], "branch": row[3], "section": row[4]}


# ---------- Academic records ----------
def insert_academic_record(student_id, subject, marks, attendance=None, exam_name=None, exam_date=None):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO academic_records (student_id, subject, marks, attendance, exam_name, exam_date, upload_batch_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (student_id, subject, marks, attendance, exam_name, exam_date, None))
    conn.commit()
    conn.close()


def insert_academic_record_with_batch(student_id, subject, marks, attendance=None, exam_name=None, exam_date=None, upload_batch_id=None):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO academic_records (student_id, subject, marks, attendance, exam_name, exam_date, upload_batch_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (student_id, subject, marks, attendance, exam_name, exam_date, upload_batch_id))
    conn.commit()
    conn.close()


def create_upload_batch(uploaded_by_user_id, filename, branches=None, years=None):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO upload_batches (uploaded_by_user_id, filename, branches, years)
        VALUES (?, ?, ?, ?)
    """, (
        uploaded_by_user_id,
        filename,
        ",".join(branches) if branches else None,
        ",".join(years) if years else None,
    ))
    conn.commit()
    batch_id = c.lastrowid
    conn.close()
    return batch_id


def finalize_upload_batch(batch_id, rows_inserted: int):
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE upload_batches SET rows_inserted = ? WHERE id = ?", (int(rows_inserted or 0), batch_id))
    conn.commit()
    conn.close()


def list_upload_batches(branch_list=None, limit=50):
    """For Principal: branch_list None returns all. For HOD: pass [branch] to restrict."""
    if branch_list is not None and len(branch_list) == 0:
        return []
    conn = get_conn()
    c = conn.cursor()
    if branch_list is None:
        c.execute("""
            SELECT id, uploaded_by_user_id, filename, branches, years, rows_inserted, created_at
            FROM upload_batches
            ORDER BY created_at DESC
            LIMIT ?
        """, (int(limit),))
        rows = c.fetchall()
        conn.close()
        return [
            {"id": r[0], "uploaded_by_user_id": r[1], "filename": r[2], "branches": r[3], "years": r[4], "rows_inserted": r[5], "created_at": r[6]}
            for r in rows
        ]

    # HOD: show only batches that include allowed branch
    b = (branch_list[0] or "").strip()
    c.execute("""
        SELECT id, uploaded_by_user_id, filename, branches, years, rows_inserted, created_at
        FROM upload_batches
        WHERE branches IS NULL OR LOWER(branches) LIKE ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (f"%{b.lower()}%", int(limit)))
    rows = c.fetchall()
    conn.close()
    return [
        {"id": r[0], "uploaded_by_user_id": r[1], "filename": r[2], "branches": r[3], "years": r[4], "rows_inserted": r[5], "created_at": r[6]}
        for r in rows
    ]


def delete_upload_batch(batch_id, branch_list=None):
    """Delete imported data for a specific upload batch.

    - Removes academic_records for that batch.
    - Removes students that become orphaned (no academic_records left).
    - Removes prediction_results for deleted students.
    """
    if branch_list is not None and len(branch_list) == 0:
        return {"ok": False, "error": "No allowed branches."}

    conn = get_conn()
    c = conn.cursor()

    # Confirm batch exists
    c.execute("SELECT id FROM upload_batches WHERE id = ?", (batch_id,))
    if not c.fetchone():
        conn.close()
        return {"ok": False, "error": "Upload batch not found."}

    # Determine affected students (respect branch restrictions for HOD)
    if branch_list is None:
        c.execute("""
            SELECT DISTINCT student_id
            FROM academic_records
            WHERE upload_batch_id = ?
        """, (batch_id,))
    else:
        placeholders = ",".join("?" * len(branch_list))
        c.execute(f"""
            SELECT DISTINCT ar.student_id
            FROM academic_records ar
            JOIN students s ON ar.student_id = s.student_id
            WHERE ar.upload_batch_id = ?
              AND s.branch IN ({placeholders})
        """, (batch_id, *branch_list))
    affected = [r[0] for r in c.fetchall()]

    # Delete the batch's academic records (respect branch restrictions for HOD)
    if branch_list is None:
        c.execute("DELETE FROM academic_records WHERE upload_batch_id = ?", (batch_id,))
        deleted_records = c.rowcount or 0
    else:
        placeholders = ",".join("?" * len(branch_list))
        c.execute(f"""
            DELETE FROM academic_records
            WHERE upload_batch_id = ?
              AND student_id IN (
                SELECT student_id FROM students WHERE branch IN ({placeholders})
              )
        """, (batch_id, *branch_list))
        deleted_records = c.rowcount or 0

    deleted_students = 0
    deleted_predictions = 0
    if affected:
        # Find orphan students (no remaining records)
        placeholders = ",".join("?" * len(affected))
        c.execute(f"""
            SELECT s.student_id
            FROM students s
            WHERE s.student_id IN ({placeholders})
              AND NOT EXISTS (
                SELECT 1 FROM academic_records ar WHERE ar.student_id = s.student_id
              )
        """, affected)
        orphans = [r[0] for r in c.fetchall()]
        if orphans:
            orphan_placeholders = ",".join("?" * len(orphans))
            c.execute(f"DELETE FROM prediction_results WHERE student_id IN ({orphan_placeholders})", orphans)
            deleted_predictions = c.rowcount or 0
            c.execute(f"DELETE FROM students WHERE student_id IN ({orphan_placeholders})", orphans)
            deleted_students = c.rowcount or 0

    # If Principal delete: remove the upload batch row itself.
    # If HOD restricted delete: keep batch row (it may contain other branches) unless it has no records remaining.
    if branch_list is None:
        c.execute("DELETE FROM upload_batches WHERE id = ?", (batch_id,))
    else:
        c.execute("SELECT 1 FROM academic_records WHERE upload_batch_id = ? LIMIT 1", (batch_id,))
        still_has_records = c.fetchone() is not None
        if not still_has_records:
            c.execute("DELETE FROM upload_batches WHERE id = ?", (batch_id,))

    conn.commit()
    conn.close()
    return {
        "ok": True,
        "deleted_records": deleted_records,
        "deleted_students": deleted_students,
        "deleted_predictions": deleted_predictions,
    }

def get_academic_records(student_ids=None, year=None, branch=None, section=None, branch_list=None):
    if student_ids is not None:
        if not student_ids:
            return []
        conn = get_conn()
        c = conn.cursor()
        placeholders = ",".join("?" * len(student_ids))
        c.execute(f"""
            SELECT ar.id, ar.student_id, s.name, s.year, s.branch, s.section, ar.subject, ar.marks, ar.attendance, ar.exam_name, ar.exam_date
            FROM academic_records ar
            JOIN students s ON ar.student_id = s.student_id
            WHERE ar.student_id IN ({placeholders})
            ORDER BY s.name, ar.subject
        """, student_ids)
        rows = c.fetchall()
        conn.close()
        return [
            {"id": r[0], "student_id": r[1], "student_name": r[2], "year": r[3], "branch": r[4], "section": r[5],
             "subject": r[6], "marks": r[7], "attendance": r[8], "exam_name": r[9], "exam_date": r[10]}
            for r in rows
        ]
    if branch_list is not None and len(branch_list) == 0:
        return []
    students = get_students_filtered(year=year, branch=branch, section=section, branch_list=branch_list)
    ids = [s["student_id"] for s in students]
    return get_academic_records(student_ids=ids)


# ---------- Prediction results ----------
def save_prediction_result(student_id, predicted_score, risk_level, analysis_summary, input_json=None):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO prediction_results (student_id, predicted_score, risk_level, analysis_summary, input_json)
        VALUES (?, ?, ?, ?, ?)
    """, (student_id, predicted_score, risk_level, analysis_summary, json.dumps(input_json) if input_json else None))
    conn.commit()
    conn.close()


def get_predictions_for_students(student_ids):
    if not student_ids:
        return []
    conn = get_conn()
    c = conn.cursor()
    placeholders = ",".join("?" * len(student_ids))
    c.execute(f"""
        SELECT pr.id, pr.student_id, s.name, pr.predicted_score, pr.risk_level, pr.analysis_summary, pr.created_at
        FROM prediction_results pr
        JOIN students s ON pr.student_id = s.student_id
        WHERE pr.student_id IN ({placeholders})
        ORDER BY pr.created_at DESC
    """, student_ids)
    rows = c.fetchall()
    conn.close()
    return [
        {"id": r[0], "student_id": r[1], "student_name": r[2], "predicted_score": r[3], "risk_level": r[4], "analysis_summary": r[5], "created_at": r[6]}
        for r in rows
    ]


# ---------- Branches / Years / Sections (distinct from data) ----------
def get_distinct_years(branch_list=None):
    if branch_list is not None and len(branch_list) == 0:
        return []
    conn = get_conn()
    c = conn.cursor()
    if branch_list is not None:
        placeholders = ",".join("?" * len(branch_list))
        c.execute(f"SELECT DISTINCT year FROM students WHERE branch IN ({placeholders}) ORDER BY year", branch_list)
    else:
        c.execute("SELECT DISTINCT year FROM students ORDER BY year")
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_distinct_branches(branch_list=None):
    if branch_list is not None and len(branch_list) == 0:
        return []
    conn = get_conn()
    c = conn.cursor()
    if branch_list is not None:
        placeholders = ",".join("?" * len(branch_list))
        c.execute(f"SELECT DISTINCT branch FROM students WHERE branch IN ({placeholders}) ORDER BY branch", branch_list)
    else:
        c.execute("SELECT DISTINCT branch FROM students ORDER BY branch")
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_distinct_sections(year=None, branch=None, branch_list=None):
    if branch_list is not None and len(branch_list) == 0:
        return []
    conn = get_conn()
    c = conn.cursor()
    q = "SELECT DISTINCT section FROM students WHERE 1=1"
    params = []
    if year:
        q += " AND year = ?"
        params.append(year)
    if branch:
        q += " AND branch = ?"
        params.append(branch)
    if branch_list is not None:
        placeholders = ",".join("?" * len(branch_list))
        q += f" AND branch IN ({placeholders})"
        params.extend(branch_list)
    q += " ORDER BY section"
    c.execute(q, params)
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]
