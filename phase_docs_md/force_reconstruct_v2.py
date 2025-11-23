import os

base_dir = r"d:\POSCO\PROCESS\phase_docs_md"
output_file = os.path.join(base_dir, "phase-draft.md")
phase_files = [f"phase-{i}.md" for i in [1, 2, 3, 3.5, 4, 5, 6, 7, 8, 9, 10]]

# Delete existing file
if os.path.exists(output_file):
    os.remove(output_file)
    print(f"Deleted {output_file}")

with open(output_file, "w", encoding="utf-8") as outfile:
    for fname in phase_files:
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as infile:
                content = infile.read()
                outfile.write(content)
                outfile.write("\n\n")
            print(f"Appended {fname}, first line: {content.splitlines()[0] if content else 'EMPTY'}")
        else:
            print(f"Warning: {fname} not found")

# Verify written content
with open(output_file, "r", encoding="utf-8") as f:
    print("First 200 chars of new file:")
    print(f.read(200))
