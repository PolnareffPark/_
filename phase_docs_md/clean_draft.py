import re

# 파일 읽기
with open('phase-draft-2.md', 'r', encoding='utf-8') as f:
    content = f.read()

print("작업 시작...")

# 1. [숫자] 형태의 참조 주석 제거
print("1. 참조 주석 제거 중...")
content = re.sub(r'\[\d+\]', '', content)

# 2. # phase-N 헤더 제거
print("2. phase 헤더 제거 중...")
content = re.sub(r'^# phase-\d+(\.\d+)?\s*$', '', content, flags=re.MULTILINE)

# 3. Phase 제목 양식 통일
print("3. Phase 제목 양식 통일 중...")
# "Phase N :" -> "[Phase N] :"
content = re.sub(r'^\[?Phase (\d+(?:\.\d+)?)\]?\s*[-–:]\s*', r'[Phase \1] : ', content, flags=re.MULTILINE)

# 특수 케이스: "Phase 3 – " 형식  
content = re.sub(r'^Phase (\d+)(\.\d+)?( – )', r'[Phase \1\2] : ', content, flags=re.MULTILINE)

# "Phase N 기술 감사" 형식
content = re.sub(r'^Phase (\d+(?:\.\d+)?) (기술 감사 보고서)', r'[Phase \1] : \2', content, flags=re.MULTILINE)

# 4. 불필요한 빈 줄 제거 (3개 이상의 연속 빈 줄을 2개로)
content = re.sub(r'\n\n\n+', '\n\n', content)

print("4. 파일 저장 중...")
# 결과 저장
with open('phase-draft-2.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ 작업 완료!")
print(f"- 참조 주석 제거 완료")
print(f"- Phase 헤더 제거 완료") 
print(f"- Phase 제목 양식 통일 완료")
print(f"- 파일 저장 완료: phase-draft-2.md")
