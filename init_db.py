# init_db.py - Edupredict AI
# Creates database tables. Register your Principal account through the app.

from database import init_db

if __name__ == "__main__":
    init_db()
    print("Edupredict AI database initialized.")
    print("Run the app and use 'Register as Principal' to create your account.")
