from cryptography.fernet import Fernet
from datetime import datetime
import os

# --- Load the same encryption key you used for logging ---
with open("secret.key", "rb") as key_file:
    encryption_key = key_file.read()

fernet = Fernet(encryption_key)

# --- Ask user which date file to decrypt ---
date_str = input("Enter date of attendance file to decrypt (YYYY-MM-DD): ").strip()

csv_file = f"{date_str}.csv"
output_file = f"{date_str}_decrypted.csv"

if not os.path.exists(csv_file):
    print(f"[ERROR] File '{csv_file}' not found in current directory.")
    exit(1)

rows = []
with open(csv_file, "rb") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith(b"gAAAAA"):  # only decrypt encrypted lines
            try:
                decrypted = fernet.decrypt(line).decode()
                rows.append(decrypted)
            except Exception as e:
                print(f"[WARNING] Could not decrypt a line: {e}")
        else:
            # skip any plaintext or malformed line
            continue

if not rows:
    print("[INFO] No decryptable lines found in this file.")
else:
    # Write all decrypted lines to a new readable CSV file
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("Name,Roll,Time\n")  # header
        for row in rows:
            out.write(row + "\n")

    print(f"[SUCCESS] Decrypted data written to '{output_file}' ✅")
