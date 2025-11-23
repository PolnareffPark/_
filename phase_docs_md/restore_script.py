import os

phase_files = [
    "phase-1.md",
    "phase-2.md",
    "phase-3.md",
    "phase-3.5.md",
    "phase-4.md",
    "phase-5.md",
    "phase-6.md",
    "phase-7.md",
    "phase-8.md",
    "phase-9.md",
    "phase-10.md"
]

output_file = "../phase-draft.md"

with open(output_file, "w", encoding="utf-8") as outfile:
    for fname in phase_files:
        if os.path.exists(fname):
            with open(fname, "r", encoding="utf-8") as infile:
                content = infile.read()
                outfile.write(content)
                outfile.write("\n\n") # Ensure separation
            print(f"Appended {fname}")
        else:
            print(f"Warning: {fname} not found")

print("Restoration complete.")
