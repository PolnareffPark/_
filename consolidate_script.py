import os
import re

directory = r'd:\POSCO\PROCESS\phase_docs_md'
output_file = os.path.join(directory, 'phase-draft.md')

phases = ['1', '2', '3', '3.5', '4', '5', '6', '7', '8', '9', '10']
filenames = [f'phase-{p}.md' for p in phases]

combined_content = ""

for fname in filenames:
    path = os.path.join(directory, fname)
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        continue
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove citation markers [1], [12]
    content = re.sub(r'\[\d+\]', '', content)
    
    # Remove bolding ** and __
    content = content.replace('**', '').replace('__', '')
    
    # Remove bullets at start of lines
    content = re.sub(r'^\s*[-*•]\s+', '', content, flags=re.MULTILINE)
    
    lines = content.split('\n')
    final_lines = []
    for line in lines:
        # Remove the "# phase-X" header line
        if re.match(r'^# phase-[\d.]+$', line.strip()):
            continue
        final_lines.append(line)
    
    # Join and strip excess whitespace
    text = '\n'.join(final_lines).strip()
    combined_content += text + "\n\n"

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(combined_content)

print(f"Successfully created {output_file} with size {len(combined_content)} chars")
