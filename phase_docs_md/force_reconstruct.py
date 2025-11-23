import os

# Absolute paths to ensure correctness
base_dir = r"d:\POSCO\PROCESS\phase_docs_md"
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
output_file = os.path.join(base_dir, "phase-draft.md")

print(f"Starting reconstruction of {output_file}...")

with open(output_file, "w", encoding="utf-8") as outfile:
    for fname in phase_files:
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as infile:
                content = infile.read()
                # Optional: Skip the first line if it's just a file header like "# phase-1"
                # But user asked to paste one by one, so let's keep it faithful to the source files for now, 
                # or maybe strip the very first line if it is just the filename.
                # Looking at the file view, line 1 is "# phase-1". Line 3 is the title.
                # I will strip the first line if it starts with "# phase-".
                lines = content.splitlines()
                if lines and lines[0].strip().lower().startswith("# phase-"):
                    content = "\n".join(lines[1:]).strip()
                
                outfile.write(content)
                outfile.write("\n\n") # Separation
            print(f"Appended {fname} (Length: {len(content)})")
        else:
            print(f"ERROR: {fname} not found at {fpath}")

print("Reconstruction finished.")
