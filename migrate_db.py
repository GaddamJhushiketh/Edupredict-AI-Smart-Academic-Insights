# migrate_db.py - Edupredict AI
# Run this once to create the new schema (edupredict_ai.db).
# Old school_system.db is not modified; use the new DB going forward.

from database import init_db, add_user

if __name__ == "__main__":
    init_db()
    add_user(
        name="Principal",
        email="principal@institution.edu",
        password="principal123",
        role="Principal",
        assigned_branch=None,
        approval_status="approved",
    )
    print("Edupredict AI database created: edupredict_ai.db")
    print("Principal: principal@institution.edu / principal123")
