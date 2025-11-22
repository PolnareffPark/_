import os
from docx import Document

# ==========================================
# 설정: 변환할 폴더 경로 (사용자 환경에 맞게 수정)
# ==========================================
SOURCE_FOLDER = './phase_drafts'  # docx 파일이 있는 폴더
OUTPUT_FOLDER = './phase_docs_md' # md 파일을 저장할 폴더

def convert_docx_to_md(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith('.docx') and not filename.startswith('~$'):
            docx_path = os.path.join(source_dir, filename)
            md_filename = os.path.splitext(filename)[0] + ".md"
            md_path = os.path.join(target_dir, md_filename)
            
            try:
                doc = Document(docx_path)
                with open(md_path, "w", encoding="utf-8") as f:
                    # 제목 처리 (파일명을 제목으로)
                    f.write(f"# {os.path.splitext(filename)[0]}\n\n")
                    
                    for para in doc.paragraphs:
                        text = para.text.strip()
                        if not text:
                            continue
                        
                        f.write(f"{text}\n\n")
                
                print(f"✅ 변환 완료: {md_filename}")
                
            except Exception as e:
                print(f"❌ 변환 실패 ({filename}): {e}")

if __name__ == "__main__":
    # pip install python-docx 필요
    convert_docx_to_md(SOURCE_FOLDER, OUTPUT_FOLDER)