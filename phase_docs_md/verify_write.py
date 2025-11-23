import os
import time

p1 = r"d:\POSCO\PROCESS\phase_docs_md\phase-1.md"
draft = r"d:\POSCO\PROCESS\phase_docs_md\phase-draft.md"

print("--- Checking phase-1.md ---")
with open(p1, "r", encoding="utf-8") as f:
    print(f.read(100))
print("\n---------------------------")

if os.path.exists(draft):
    try:
        os.remove(draft)
        print(f"Deleted {draft}")
    except Exception as e:
        print(f"Failed to delete {draft}: {e}")

time.sleep(1)

if os.path.exists(draft):
    print("ERROR: File still exists after deletion!")
else:
    print("File successfully deleted.")

print("--- Writing to phase-draft.md ---")
with open(p1, "r", encoding="utf-8") as infile:
    content = infile.read()

with open(draft, "w", encoding="utf-8") as outfile:
    outfile.write(content)

print("--- Reading back phase-draft.md ---")
with open(draft, "r", encoding="utf-8") as f:
    print(f.read(100))
print("\n---------------------------")
