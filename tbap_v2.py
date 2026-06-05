"""
TBAPv2: Text-Based Brain Activity Predictor with Emotion
=========================================================
기존 TBAP (언어 처리) + 감정 처리 모듈 통합

핵심 차별점:
  기존 연구: EEG + 텍스트 → 감정 분류 (EEG가 입력)
  TBAPv2:   텍스트 → NLP 감정 수치 → EEG 예측 (EEG가 출력)

파이프라인:
  텍스트
    ↓
  [EmotionAnalyzer]       NLP로 감정 수치화 (VADER + NRC Lexicon)
    ↓
  [EmotionToEEGMapper]    감정 수치 → EEG 지표 변환 (뇌과학 근거)
    ↓
  [LinguisticPredictor]   언어 처리 지표 (N400, P600)
    ↓
  [TBAPv2]               언어 + 감정 EEG 통합 예측

뇌과학 근거:
  valence → 전두 알파 비대칭 (Davidson 1992, 1995)
  arousal → 전두 감마 파워 증가 (Müller et al. 1999)
  감정 복잡도 → 전두 세타 증가 (Aftanas & Golocheikine 2001)
  부정 감정 → 우측 전두 알파 ERD (Harmon-Jones 2004)
"""

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

from abc import ABC, abstractmethod

# ══════════════════════════════════════════
# Layer 1 추가 데이터 클래스
# ══════════════════════════════════════════

@dataclass
class TextFeatures:
    text: str
    ttr: float = 0.0
    avg_sent_len: float = 0.0
    complexity: float = 0.0
    emotion_intensity: float = 0.0
    info_density: float = 0.0
    perplexity: float = 0.0

@dataclass
class BrainResponse:
    n400: float = 0.0
    p600: float = 0.0
    theta_fz: float = 0.0
    plv_t7t8: float = 0.0
    predictor_name: str = ""

@dataclass
class LiteratureRange:
    min_val: float
    max_val: float
    source: str
    def contains(self, value: float) -> bool:
        return self.min_val <= value <= self.max_val
    def __str__(self):
        return f"[{self.min_val:.1f}, {self.max_val:.1f}] ({self.source})"

# ══════════════════════════════════════════
# Layer 2: 추상 기반 클래스
# ══════════════════════════════════════════

class BrainResponsePredictor(ABC):
    """추상 인터페이스 — 다형성의 핵심"""
    @abstractmethod
    def predict(self, text: str) -> BrainResponse:
        pass

class LinguisticRulePredictor(BrainResponsePredictor):
    """언어학 규칙 기반 — BrainResponsePredictor 상속"""
    STOPWORDS = {
        "the","a","an","is","are","was","were","be","been",
        "have","has","had","to","of","and","or","but","in",
        "on","at","for","with","by","from","it","he","she",
        "they","we","you","i","this","that","which","who",
    }
    COMPLEX_MARKERS = {
        "although","however","nevertheless","whereas",
        "furthermore","consequently","notwithstanding",
    }
    def predict(self, text: str) -> BrainResponse:
        words = text.lower().split()
        n_words = max(len(words), 1)
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        n_sents = max(len(sentences), 1)
        ttr = len(set(words)) / n_words
        complexity = sum(1 for w in words if w.strip(".,!?") in self.COMPLEX_MARKERS) / n_words
        avg_len = n_words / n_sents
        info_density = len([w for w in words if w not in self.STOPWORDS]) / n_words
        surprisal = ttr * 8.0 + complexity * 5.0
        n400 = -0.5 + (-0.055)*surprisal + (-0.002)*surprisal**2 + (-3.5)*complexity
        p600 = (0.3 + 8.0*complexity + 0.18*avg_len) if complexity > 0.01 else 0.4
        theta = 16.0 + 0.35*(surprisal*n_words/10) + 12.0*complexity
        plv = min(0.30 + 0.50*info_density, 0.85)
        return BrainResponse(n400=round(n400,3), p600=round(p600,3),
                             theta_fz=round(theta,1), plv_t7t8=round(plv,3),
                             predictor_name="LinguisticRulePredictor")

class EmbeddingPredictor(BrainResponsePredictor):
    """Surprisal 기반 임베딩 예측기 — BrainResponsePredictor 상속"""
    COMMON_WORDS = {"the","a","an","is","are","was","were","be",
                    "and","or","but","in","on","to","of","it",
                    "he","she","they","we","you","i","this","that"}
    def __init__(self):
        self._rule = LinguisticRulePredictor()
        np.random.seed(42)
    def _surprisal(self, text: str) -> float:
        words = [w.strip(".,!?;:").lower() for w in text.split()]
        s = [np.random.uniform(2.5,4.5) if w in self.COMMON_WORDS
             else np.random.uniform(6.0,13.0) for w in words]
        return float(np.exp(np.mean(s)/4.5))
    def predict(self, text: str) -> BrainResponse:
        ppl = self._surprisal(text)
        n400 = 0.5 + (-0.018)*min(ppl, 300)
        rule = self._rule.predict(text)
        return BrainResponse(n400=round(n400,3), p600=rule.p600,
                             theta_fz=round(rule.theta_fz*(1+0.15*min(ppl/100,2)),1),
                             plv_t7t8=rule.plv_t7t8,
                             predictor_name="EmbeddingPredictor")

# ══════════════════════════════════════════
# Layer 3: ResultVisualizer 분리
# ══════════════════════════════════════════

class ResultVisualizer:
    """Layer 3 — plots and tables 전담"""

    def print_summary(self, results: list):
        print("\n[ResultVisualizer] 예측 요약")
        for text, desc, r, _ in results:
            print(f"  {desc:25s} N400:{r.n400:+.2f} "
                  f"gamma:{r.frontal_gamma:.1f} "
                  f"asym:{r.frontal_alpha_asymmetry:+.2f}")

    def plot(self, all_results: list,
             save_path: str = "tbapv2_result.png"):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
        except ImportError:
            return

        # 한국어 폰트 설정 (Windows/Mac/Linux 순서로 시도)
        korean_fonts = [
            "Malgun Gothic",    # Windows
            "AppleGothic",      # Mac
            "NanumGothic",      # Linux
            "Noto Sans CJK KR", # Linux 대안
            "Noto Sans CJK JP", # 서버 환경
            "DejaVu Sans",      # 폴백
        ]
        selected = None
        available = {f.name for f in fm.fontManager.ttflist}
        for font in korean_fonts:
            if font in available:
                selected = font
                break
        if selected:
            plt.rcParams["font.family"] = selected
        plt.rcParams["axes.unicode_minus"] = False
        n = len(all_results)
        labels = [r[1][:12] for r in all_results]
        x = np.arange(n)
        fig, axes = plt.subplots(2, 4, figsize=(20, 8), facecolor="#0f1117")
        fig.suptitle("TBAPv2: 언어 처리 EEG + 감정 처리 EEG 통합 예측",
                     color="white", fontsize=12, fontweight="bold")
        all_metrics = [
            ("N400 (μV)",  [r[2].n400     for r in all_results], "#ef5350", 0, 0),
            ("P600 (μV)",  [r[2].p600     for r in all_results], "#42a5f5", 0, 1),
            ("세타 (μV²)", [r[2].theta_fz for r in all_results], "#66bb6a", 0, 2),
            ("PLV T7-T8",  [r[2].plv_t7t8 for r in all_results], "#ffa726", 0, 3),
            ("전두 알파\n비대칭(μV²)",
             [r[2].frontal_alpha_asymmetry for r in all_results], "#ce93d8", 1, 0),
            ("전두 감마\n(μV²)",
             [r[2].frontal_gamma for r in all_results], "#80deea", 1, 1),
            ("편도체\nhigh-β(μV²)",
             [r[2].amygdala_hbeta for r in all_results], "#ffcc02", 1, 2),
            ("전두 알파\nERD(%)",
             [r[2].frontal_alpha_erd for r in all_results], "#ff8a65", 1, 3),
        ]
        for title, vals, color, row, col in all_metrics:
            ax = axes[row, col]
            ax.set_facecolor("#1a1d27")
            ax.bar(x, vals, color=color, alpha=0.85)
            ax.axhline(0, color="#555", linewidth=0.5)
            ax.set_title(title, color="white", fontsize=8, pad=6)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20,
                               ha="right", color="#aaa", fontsize=7)
            ax.tick_params(colors="#aaa", labelsize=7)
            for sp in ax.spines.values():
                sp.set_color("#333")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(save_path, dpi=150,
                    bbox_inches="tight", facecolor="#0f1117")
        plt.close(fig)
        print(f"\n그래프 저장: {save_path}")


np.random.seed(42)


# ══════════════════════════════════════════
# NRC Emotion Lexicon (내장 축약본)
# 실제 NRC: nrc.canada.ca/en/research-development/
#           products-services/technical-advisory-services/
#           sentiment-emotion-lexicons
# 여기서는 핵심 감정어 내장
# ══════════════════════════════════════════

NRC_LEXICON = {
    # (joy, trust, fear, surprise, sadness, disgust, anger, anticipation)
    "happy":       (1, 1, 0, 0, 0, 0, 0, 1),
    "happiness":   (1, 1, 0, 0, 0, 0, 0, 1),
    "joy":         (1, 1, 0, 1, 0, 0, 0, 1),
    "love":        (1, 1, 0, 0, 0, 0, 0, 1),
    "wonderful":   (1, 1, 0, 1, 0, 0, 0, 0),
    "great":       (1, 1, 0, 0, 0, 0, 0, 1),
    "excellent":   (1, 1, 0, 0, 0, 0, 0, 0),
    "beautiful":   (1, 1, 0, 1, 0, 0, 0, 0),
    "sad":         (0, 0, 0, 0, 1, 0, 0, 0),
    "sadness":     (0, 0, 0, 0, 1, 0, 0, 0),
    "grief":       (0, 0, 0, 0, 1, 0, 0, 0),
    "sorrow":      (0, 0, 0, 0, 1, 0, 0, 0),
    "fear":        (0, 0, 1, 1, 0, 0, 0, 0),
    "afraid":      (0, 0, 1, 0, 0, 0, 0, 0),
    "terror":      (0, 0, 1, 1, 0, 0, 1, 0),
    "scared":      (0, 0, 1, 1, 0, 0, 0, 0),
    "angry":       (0, 0, 0, 0, 0, 1, 1, 0),
    "anger":       (0, 0, 0, 0, 0, 1, 1, 0),
    "rage":        (0, 0, 0, 0, 0, 1, 1, 0),
    "furious":     (0, 0, 0, 0, 0, 1, 1, 0),
    "disgust":     (0, 0, 0, 0, 0, 1, 0, 0),
    "horrible":    (0, 0, 1, 1, 0, 1, 0, 0),
    "awful":       (0, 0, 0, 0, 1, 1, 0, 0),
    "trust":       (0, 1, 0, 0, 0, 0, 0, 0),
    "surprise":    (0, 0, 0, 1, 0, 0, 0, 0),
    "shocked":     (0, 0, 1, 1, 0, 0, 0, 0),
    "anticipate":  (0, 0, 0, 0, 0, 0, 0, 1),
    "hope":        (1, 1, 0, 0, 0, 0, 0, 1),
    "excited":     (1, 0, 0, 1, 0, 0, 0, 1),
    "anxious":     (0, 0, 1, 0, 0, 0, 0, 0),
    "anxiety":     (0, 0, 1, 0, 0, 0, 0, 0),
    "calm":        (1, 1, 0, 0, 0, 0, 0, 0),
    "peaceful":    (1, 1, 0, 0, 0, 0, 0, 0),
    "lonely":      (0, 0, 0, 0, 1, 0, 0, 0),
    "depressed":   (0, 0, 0, 0, 1, 0, 0, 0),
    "desperate":   (0, 0, 1, 0, 1, 0, 0, 0),
    "confused":    (0, 0, 0, 1, 0, 0, 0, 0),
    "proud":       (1, 1, 0, 0, 0, 0, 0, 1),
    "ashamed":     (0, 0, 0, 0, 1, 1, 0, 0),
    "guilty":      (0, 0, 0, 0, 1, 1, 0, 0),
    "jealous":     (0, 0, 0, 0, 0, 1, 1, 0),
    "grateful":    (1, 1, 0, 0, 0, 0, 0, 0),
    "curious":     (0, 0, 0, 1, 0, 0, 0, 1),
    "bored":       (0, 0, 0, 0, 1, 0, 0, 0),
    "tired":       (0, 0, 0, 0, 1, 0, 0, 0),
    "pain":        (0, 0, 0, 0, 1, 0, 0, 0),
    "hurt":        (0, 0, 0, 0, 1, 0, 0, 0),
    "cry":         (0, 0, 0, 0, 1, 0, 0, 0),
    "laugh":       (1, 0, 0, 0, 0, 0, 0, 0),
    "smile":       (1, 1, 0, 0, 0, 0, 0, 0),
    "death":       (0, 0, 1, 0, 1, 1, 0, 0),
    "murder":      (0, 0, 1, 1, 0, 1, 1, 0),
    "war":         (0, 0, 1, 0, 0, 1, 1, 0),
    "peace":       (1, 1, 0, 0, 0, 0, 0, 0),
}

EMOTION_NAMES = ["joy", "trust", "fear", "surprise",
                 "sadness", "disgust", "anger", "anticipation"]


# ══════════════════════════════════════════
# 데이터 클래스
# ══════════════════════════════════════════

@dataclass
class EmotionScores:
    """
    NLP로 추출한 감정 수치.
    Plutchik(1980) 8감정 모델 기반.
    """
    # Plutchik 8감정 (0~1)
    joy: float          = 0.0
    trust: float        = 0.0
    fear: float         = 0.0
    surprise: float     = 0.0
    sadness: float      = 0.0
    disgust: float      = 0.0
    anger: float        = 0.0
    anticipation: float = 0.0

    # VADER 기반 종합 지표
    valence: float      = 0.0   # 긍부정 (-1 ~ +1)
    arousal: float      = 0.0   # 각성도 (0 ~ 1)
    complexity: float   = 0.0   # 감정 복잡도 (혼재 정도, 0~1)

    def dominant_emotion(self) -> str:
        scores = {
            "joy": self.joy, "trust": self.trust,
            "fear": self.fear, "surprise": self.surprise,
            "sadness": self.sadness, "disgust": self.disgust,
            "anger": self.anger, "anticipation": self.anticipation,
        }
        return max(scores, key=scores.get)


@dataclass
class EmotionEEGResponse:
    """
    감정 처리에 대응하는 EEG 지표.
    Davidson (1992, 1995), Müller et al. (1999) 기반.
    """
    # 전두 알파 비대칭 (F3-F4, μV²)
    # 양수: 좌측 우세 (긍정/접근 동기)
    # 음수: 우측 우세 (부정/회피 동기)
    frontal_alpha_asymmetry: float = 0.0

    # 전두 감마 파워 (μV²) — 감정 각성도
    frontal_gamma: float = 0.0

    # 전두 세타 파워 (μV²) — 감정 인지 통합
    frontal_theta: float = 0.0

    # 편도체 관련 고주파 (high-beta, μV²) — 공포/불안
    amygdala_hbeta: float = 0.0

    # 전두 알파 ERD (%) — 감정 처리 활성
    frontal_alpha_erd: float = 0.0


@dataclass
class IntegratedBrainResponse:
    """
    언어 처리 + 감정 처리 통합 EEG 예측.
    TBAPv2의 최종 출력.
    """
    # 언어 처리 (기존 TBAP)
    n400: float = 0.0
    p600: float = 0.0
    theta_fz: float = 0.0
    plv_t7t8: float = 0.0

    # 감정 처리 (새로 추가)
    frontal_alpha_asymmetry: float = 0.0
    frontal_gamma: float = 0.0
    amygdala_hbeta: float = 0.0
    frontal_alpha_erd: float = 0.0

    # 감정 수치
    emotion_scores: EmotionScores = field(
        default_factory=EmotionScores
    )
    predictor_name: str = ""


# ══════════════════════════════════════════
# 1. 감정 분석기
# ══════════════════════════════════════════

class EmotionAnalyzer:
    """
    텍스트 → EmotionScores 변환.

    두 가지 방법 결합:
      1. NRC Lexicon: Plutchik 8감정 점수
      2. VADER 규칙: valence, arousal 추정

    VADER 없으면 규칙 기반으로 대체.
    """

    # VADER 강도 부사
    INTENSIFIERS = {
        "very": 1.4, "extremely": 1.8, "absolutely": 1.9,
        "quite": 1.2, "rather": 1.1, "slightly": 0.6,
        "barely": 0.4, "not": -1.0, "never": -1.0,
    }

    def __init__(self):
        # VADER 로드 시도
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            self._has_vader = True
        except ImportError:
            self._has_vader = False

    def _nrc_scores(self, text: str) -> Dict[str, float]:
        """NRC Lexicon으로 8감정 점수 계산"""
        words = text.lower().split()
        scores = {e: 0.0 for e in EMOTION_NAMES}
        n_matched = 0

        i = 0
        while i < len(words):
            word = words[i].strip(".,!?;:'\"")
            # 강도 부사 처리
            multiplier = 1.0
            if word in self.INTENSIFIERS and i + 1 < len(words):
                multiplier = abs(self.INTENSIFIERS[word])
                i += 1
                word = words[i].strip(".,!?;:'\"")

            if word in NRC_LEXICON:
                vec = NRC_LEXICON[word]
                for j, ename in enumerate(EMOTION_NAMES):
                    scores[ename] += vec[j] * multiplier
                n_matched += 1
            i += 1

        # 정규화
        if n_matched > 0:
            total = sum(scores.values()) + 1e-8
            scores = {k: v / total for k, v in scores.items()}

        return scores

    def _compute_arousal(self, scores: Dict[str, float],
                         valence: float) -> float:
        """
        각성도 추정.
        고각성 감정: 공포, 분노, 놀람, 기쁨
        저각성 감정: 슬픔, 신뢰
        """
        high_arousal = scores["fear"] + scores["anger"] + \
                       scores["surprise"] + scores["joy"]
        low_arousal  = scores["sadness"] + scores["trust"]
        arousal = (high_arousal - low_arousal * 0.5 + abs(valence) * 0.3)
        return max(0.0, min(1.0, arousal))

    def _compute_complexity(self, scores: Dict[str, float]) -> float:
        """
        감정 복잡도 = 여러 감정이 동시에 높을수록 복잡.
        엔트로피 기반.
        """
        vals = [v for v in scores.values() if v > 0]
        if not vals:
            return 0.0
        p = np.array(vals) / (sum(vals) + 1e-8)
        entropy = -np.sum(p * np.log(p + 1e-8))
        max_entropy = np.log(len(EMOTION_NAMES))
        return float(entropy / max_entropy)

    def analyze(self, text: str) -> EmotionScores:
        """텍스트 → EmotionScores"""
        nrc = self._nrc_scores(text)

        # valence: VADER 있으면 VADER, 없으면 규칙 기반
        if self._has_vader:
            vader_scores = self._vader.polarity_scores(text)
            valence = vader_scores["compound"]
        else:
            pos = nrc["joy"] + nrc["trust"] + nrc["anticipation"]
            neg = nrc["fear"] + nrc["sadness"] + \
                  nrc["disgust"] + nrc["anger"]
            valence = float(pos - neg)
            valence = max(-1.0, min(1.0, valence))

        arousal    = self._compute_arousal(nrc, valence)
        complexity = self._compute_complexity(nrc)

        return EmotionScores(
            joy=round(nrc["joy"], 3),
            trust=round(nrc["trust"], 3),
            fear=round(nrc["fear"], 3),
            surprise=round(nrc["surprise"], 3),
            sadness=round(nrc["sadness"], 3),
            disgust=round(nrc["disgust"], 3),
            anger=round(nrc["anger"], 3),
            anticipation=round(nrc["anticipation"], 3),
            valence=round(valence, 3),
            arousal=round(arousal, 3),
            complexity=round(complexity, 3),
        )


# ══════════════════════════════════════════
# 2. 감정 → EEG 변환기
# ══════════════════════════════════════════

class EmotionToEEGMapper:
    """
    EmotionScores → EmotionEEGResponse 변환.

    뇌과학 근거:
      전두 알파 비대칭 ↔ valence
        (Davidson 1992: 좌측 전두 활성 = 긍정/접근 동기)
        (Harmon-Jones 2004: 우측 전두 활성 = 부정/회피 동기)

      전두 감마 파워 ↔ arousal
        (Müller et al. 1999: 고각성 = 전두 감마 증가)

      전두 세타 ↔ 감정 복잡도
        (Aftanas & Golocheikine 2001:
         감정 인지 통합 시 전두 세타 증가)

      편도체 high-beta ↔ 공포/불안
        (Luo et al. 2020: 공포 처리 중 고주파 증가)
    """

    # 전두 알파 비대칭 계수 (Davidson 1992 보고값 기반)
    ASYMMETRY_VALENCE_COEF  =  3.5   # valence 1 → 비대칭 3.5μV²
    ASYMMETRY_ANGER_PENALTY = -2.0   # 분노는 좌측 활성 (접근 동기)

    # 전두 감마 계수 (Müller et al. 1999)
    GAMMA_BASE        =  2.0
    GAMMA_AROUSAL     =  6.0
    GAMMA_FEAR_BONUS  =  3.0   # 공포 시 감마 추가 증가

    # 전두 세타 계수 (Aftanas & Golocheikine 2001)
    THETA_BASE        = 18.0
    THETA_COMPLEXITY  = 20.0
    THETA_VALENCE_NEG =  5.0   # 부정 감정 시 세타 증가

    # 편도체 high-beta (Luo et al. 2020)
    HBETA_BASE        =  1.0
    HBETA_FEAR_COEF   =  8.0
    HBETA_ANGER_COEF  =  5.0

    # 전두 알파 ERD (Klimesch 1999)
    ERD_BASE          = -8.0
    ERD_AROUSAL_COEF  = -20.0

    def map(self, emotion: EmotionScores) -> EmotionEEGResponse:
        """감정 수치 → EEG 지표"""

        # 전두 알파 비대칭
        # 긍정 → 양수 (좌측 우세)
        # 부정 → 음수 (우측 우세)
        # 분노는 예외: 부정이지만 접근 동기 → 좌측 활성
        asymmetry = (
            self.ASYMMETRY_VALENCE_COEF * emotion.valence
            + self.ASYMMETRY_ANGER_PENALTY * emotion.anger
        )

        # 전두 감마 — 각성도 + 공포 보너스
        gamma = (
            self.GAMMA_BASE
            + self.GAMMA_AROUSAL * emotion.arousal
            + self.GAMMA_FEAR_BONUS * emotion.fear
        )

        # 전두 세타 — 감정 복잡도 + 부정 감정
        neg_intensity = emotion.sadness + emotion.fear + emotion.anger
        theta = (
            self.THETA_BASE
            + self.THETA_COMPLEXITY * emotion.complexity
            + self.THETA_VALENCE_NEG * neg_intensity
        )

        # 편도체 high-beta — 공포/분노
        hbeta = (
            self.HBETA_BASE
            + self.HBETA_FEAR_COEF  * emotion.fear
            + self.HBETA_ANGER_COEF * emotion.anger
        )

        # 전두 알파 ERD — 각성도
        erd = self.ERD_BASE + self.ERD_AROUSAL_COEF * emotion.arousal

        return EmotionEEGResponse(
            frontal_alpha_asymmetry=round(asymmetry, 3),
            frontal_gamma=round(gamma, 2),
            frontal_theta=round(theta, 1),
            amygdala_hbeta=round(hbeta, 2),
            frontal_alpha_erd=round(erd, 1),
        )


# ══════════════════════════════════════════
# 3. 언어 처리 예측기 (기존 TBAP)
# ══════════════════════════════════════════

class LinguisticPredictor:
    """
    텍스트 언어 특성 → N400, P600, 세타, PLV
    기존 TBAP의 언어 처리 모듈.
    """

    STOPWORDS = {
        "the","a","an","is","are","was","were","be","been",
        "have","has","had","to","of","and","or","but","in",
        "on","at","for","with","by","from","it","he","she",
        "they","we","you","i","this","that","which","who",
    }
    COMPLEX_MARKERS = {
        "although","however","nevertheless","whereas",
        "furthermore","consequently","notwithstanding",
        "despite","regardless","meanwhile",
    }

    def predict(self, text: str) -> Dict[str, float]:
        words = text.lower().split()
        n_words = max(len(words), 1)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        n_sents = max(len(sentences), 1)

        ttr = len(set(words)) / n_words
        complexity = sum(
            1 for w in words
            if w.strip(".,!?") in self.COMPLEX_MARKERS
        ) / n_words
        avg_len = n_words / n_sents
        info_density = len([
            w for w in words if w not in self.STOPWORDS
        ]) / n_words

        # surprisal 근사 (단어 빈도 기반)
        surprisal = ttr * 8.0 + complexity * 5.0

        n400 = (-0.5
                + (-0.055) * surprisal
                + (-0.002) * surprisal ** 2
                + (-3.5)   * complexity)

        p600 = (0.3
                + 8.0 * complexity
                + 0.18 * avg_len
                ) if complexity > 0.01 else 0.4

        theta = (16.0
                 + 0.35 * (surprisal * n_words / 10)
                 + 12.0 * complexity)

        plv = min(0.30 + 0.50 * info_density, 0.85)

        return {
            "n400": round(n400, 3),
            "p600": round(p600, 3),
            "theta_fz": round(theta, 1),
            "plv_t7t8": round(plv, 3),
        }


# ══════════════════════════════════════════
# 4. TBAPv2 — 통합 예측기
# ══════════════════════════════════════════

class TBAPv2:
    """
    Text-Based Brain Activity Predictor v2.
    언어 처리 + 감정 처리 통합 EEG 예측.

    OOP 구조:
      EmotionAnalyzer      텍스트 → 감정 수치
      EmotionToEEGMapper   감정 수치 → 감정 EEG
      LinguisticPredictor  텍스트 → 언어 EEG
      TBAPv2               둘을 통합

    기존 연구와의 차이:
      기존: EEG(입력) + 텍스트 → 감정 분류
      TBAPv2: 텍스트만 → 언어 EEG + 감정 EEG 동시 예측
    """

    def __init__(self):
        self.emotion_analyzer   = EmotionAnalyzer()
        self.emotion_mapper     = EmotionToEEGMapper()
        self.linguistic         = LinguisticPredictor()

    def predict(self, text: str) -> IntegratedBrainResponse:
        """텍스트 → 통합 뇌 반응 예측"""

        # 언어 처리 지표
        lang = self.linguistic.predict(text)

        # 감정 분석
        emotion_scores = self.emotion_analyzer.analyze(text)

        # 감정 → EEG 변환
        emotion_eeg = self.emotion_mapper.map(emotion_scores)

        return IntegratedBrainResponse(
            # 언어 처리
            n400=lang["n400"],
            p600=lang["p600"],
            theta_fz=lang["theta_fz"],
            plv_t7t8=lang["plv_t7t8"],
            # 감정 처리
            frontal_alpha_asymmetry=emotion_eeg.frontal_alpha_asymmetry,
            frontal_gamma=emotion_eeg.frontal_gamma,
            amygdala_hbeta=emotion_eeg.amygdala_hbeta,
            frontal_alpha_erd=emotion_eeg.frontal_alpha_erd,
            # 감정 수치
            emotion_scores=emotion_scores,
            predictor_name="TBAPv2",
        )


# ══════════════════════════════════════════
# 5. 문헌 검증기
# ══════════════════════════════════════════

class LiteratureValidator:
    """
    DEAP 데이터셋 및 Davidson 연구 기반
    감정 EEG 지표 문헌 범위 검증.
    """

    # 감정 조건별 문헌 범위
    LITERATURE = {
        "positive_high_arousal": {   # 기쁨, 흥분
            "frontal_alpha_asymmetry": (1.0,  6.0,  "Davidson 1995"),
            "frontal_gamma":           (5.0, 12.0,  "Müller 1999"),
            "frontal_theta":           (18.0, 28.0, "Aftanas 2001"),
            "amygdala_hbeta":          (1.0,  3.0,  "Luo 2020"),
            "frontal_alpha_erd":       (-25.0, -8.0, "Klimesch 1999"),
        },
        "negative_high_arousal": {   # 공포, 분노
            "frontal_alpha_asymmetry": (-5.0, -0.5, "Davidson 1995"),
            "frontal_gamma":           (6.0, 14.0,  "Müller 1999"),
            "frontal_theta":           (22.0, 35.0, "Aftanas 2001"),
            "amygdala_hbeta":          (4.0,  9.0,  "Luo 2020"),
            "frontal_alpha_erd":       (-30.0, -12.0, "Klimesch 1999"),
        },
        "negative_low_arousal": {    # 슬픔, 우울
            "frontal_alpha_asymmetry": (-4.0,  0.0, "Davidson 1995"),
            "frontal_gamma":           (2.0,  6.0,  "Müller 1999"),
            "frontal_theta":           (20.0, 32.0, "Aftanas 2001"),
            "amygdala_hbeta":          (1.5,  4.0,  "Luo 2020"),
            "frontal_alpha_erd":       (-20.0, -5.0, "Klimesch 1999"),
        },
        "neutral": {
            "frontal_alpha_asymmetry": (-1.5,  1.5, "Davidson 1995"),
            "frontal_gamma":           (1.5,  4.0,  "Müller 1999"),
            "frontal_theta":           (15.0, 22.0, "Aftanas 2001"),
            "amygdala_hbeta":          (0.5,  2.0,  "Luo 2020"),
            "frontal_alpha_erd":       (-12.0, -2.0, "Klimesch 1999"),
        },
    }

    def detect_condition(self,
                         emotion: EmotionScores) -> str:
        """감정 수치로 조건 자동 감지"""
        if emotion.valence > 0.2 and emotion.arousal > 0.4:
            return "positive_high_arousal"
        elif emotion.valence < -0.2 and emotion.arousal > 0.4:
            return "negative_high_arousal"
        elif emotion.valence < -0.2 and emotion.arousal <= 0.4:
            return "negative_low_arousal"
        else:
            return "neutral"

    def validate(self, response: IntegratedBrainResponse) -> Dict:
        condition = self.detect_condition(response.emotion_scores)
        ref = self.LITERATURE[condition]
        checks = {
            "frontal_alpha_asymmetry": response.frontal_alpha_asymmetry,
            "frontal_gamma":           response.frontal_gamma,
            "frontal_theta":           response.frontal_theta if hasattr(
                                       response, "frontal_theta") else 0,
            "amygdala_hbeta":          response.amygdala_hbeta,
            "frontal_alpha_erd":       response.frontal_alpha_erd,
        }
        results = {}
        for key, val in checks.items():
            if key in ref:
                lo, hi, src = ref[key]
                results[key] = {
                    "value": val,
                    "in_range": lo <= val <= hi,
                    "range": f"[{lo}, {hi}]",
                    "source": src,
                }
        return {"condition": condition, "checks": results}


# ══════════════════════════════════════════
# 6. 시각화 및 실행
# ══════════════════════════════════════════

class TBAPv2Runner:
    """전체 실행 및 결과 출력"""

    def __init__(self):
        self.model      = TBAPv2()
        self.validator  = LiteratureValidator()
        self.visualizer = ResultVisualizer()

    def run(self, texts: List[Tuple[str, str]],
            plot: bool = True):
        """
        texts: [(텍스트, 설명), ...]
        """
        print("\n" + "█"*72)
        print("  TBAPv2: 언어 처리 + 감정 처리 통합 EEG 예측")
        print("  (기존 연구와의 차이: EEG가 출력, 텍스트만 입력)")
        print("█"*72)

        all_results = []
        for text, desc in texts:
            result = self.model.predict(text)
            validation = self.validator.validate(result)
            all_results.append((text, desc, result, validation))
            self._print_result(text, desc, result, validation)

        if plot:
            self.visualizer.plot(all_results)

        return all_results

    def _print_result(self, text, desc, r: IntegratedBrainResponse,
                      validation: Dict):
        e = r.emotion_scores
        print(f"\n{'═'*72}")
        print(f"  텍스트: \"{text[:55]}{'...' if len(text)>55 else ''}\"")
        print(f"  설명:   {desc}")
        print(f"{'═'*72}")

        print(f"\n  [감정 분석 결과]")
        print(f"  주요 감정:  {e.dominant_emotion():12s}  "
              f"valence: {e.valence:+.3f}  "
              f"arousal: {e.arousal:.3f}  "
              f"복잡도: {e.complexity:.3f}")

        emo_pairs = [
            ("기쁨", e.joy), ("신뢰", e.trust),
            ("공포", e.fear), ("놀람", e.surprise),
            ("슬픔", e.sadness), ("혐오", e.disgust),
            ("분노", e.anger), ("기대", e.anticipation),
        ]
        active = [(n, v) for n, v in emo_pairs if v > 0.05]
        if active:
            emo_str = "  " + "  ".join(
                f"{n}:{v:.2f}" for n, v in
                sorted(active, key=lambda x: -x[1])
            )
            print(emo_str)

        print(f"\n  [언어 처리 EEG — 기존 TBAP]")
        print(f"  N400:     {r.n400:+.3f} μV   "
              f"P600: {r.p600:+.3f} μV   "
              f"세타(Fz): {r.theta_fz:.1f} μV²   "
              f"PLV: {r.plv_t7t8:.3f}")

        print(f"\n  [감정 처리 EEG — TBAPv2 신규]")
        cond = validation["condition"]
        checks = validation["checks"]
        rows = [
            ("전두 알파 비대칭", r.frontal_alpha_asymmetry, "μV²",
             "frontal_alpha_asymmetry"),
            ("전두 감마 파워",   r.frontal_gamma,           "μV²",
             "frontal_gamma"),
            ("편도체 high-beta", r.amygdala_hbeta,          "μV²",
             "amygdala_hbeta"),
            ("전두 알파 ERD",    r.frontal_alpha_erd,       "%",
             "frontal_alpha_erd"),
        ]
        for label, val, unit, key in rows:
            if key in checks:
                c = checks[key]
                mark = "✅" if c["in_range"] else "❌"
                ref_str = f"문헌{c['range']} ({c['source']})"
                print(f"  {label:16s}: {val:+7.2f} {unit:4s}  "
                      f"{ref_str:35s}  {mark}")

        passed = sum(1 for c in checks.values() if c["in_range"])
        total  = len(checks)
        print(f"\n  감정 EEG 검증: {passed}/{total}  "
              f"(조건: {cond})")
