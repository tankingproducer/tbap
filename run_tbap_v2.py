"""
TBAPv2 실행 스크립트
--------------------
실행 방법:
    python run_tbap_v2.py

필요 패키지:
    pip install vaderSentiment numpy matplotlib
"""
from tbap_v2 import TBAPv2Runner

runner = TBAPv2Runner()

texts = [
    # 중립 — 감정 없음
    ("The dog chased the ball in the park.",
     "중립 문장"),

    # 긍정 고각성 — 기쁨, 흥분
    ("I am so excited and happy about this wonderful news!",
     "긍정 고각성 (기쁨/흥분)"),

    # 부정 고각성 — 공포, 분노
    ("The terror and rage I felt was absolutely overwhelming.",
     "부정 고각성 (공포/분노)"),

    # 부정 저각성 — 슬픔
    ("I feel so sad and lonely. The grief never goes away.",
     "부정 저각성 (슬픔)"),

    # 복합 감정 — 여러 감정 혼재
    ("Although I was grateful, the fear and sadness made me confused.",
     "복합 감정 + 문법 복잡"),
]

results = runner.run(texts, plot=True)
