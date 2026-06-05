# TBAP V2

이 저장소에는 `tbap_v2.py`와 `run_tbap_v2.py` 파일이 포함되어 있습니다.

## 실행 방법

1. Python이 설치되어 있는지 확인합니다.
   - Windows: `python --version`
   - macOS/Linux: `python3 --version`

2. 필요한 경우 가상 환경을 생성합니다.
   ```bash
   python -m venv venv
   # 또는 python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```

3. 필요한 패키지를 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```

4. `run_tbap_v2.py`를 실행하여 전체 스크립트를 실행합니다.
   ```bash
   python run_tbap_v2.py
   ```

5. `tbap_v2.py`를 직접 실행하려면:
   ```bash
   python tbap_v2.py
   ```

## 파일 설명

- `run_tbap_v2.py`: 메인 실행 스크립트입니다.
- `tbap_v2.py`: 실제 처리 로직을 포함하는 모듈 파일입니다.

## 참고

- 이 문서는 Windows와 macOS/Linux 환경 모두에서 실행 방법을 안내합니다.
- 추가 패키지 또는 의존성이 있을 경우 `requirements.txt` 파일을 만들어 관리하는 것이 좋습니다.
