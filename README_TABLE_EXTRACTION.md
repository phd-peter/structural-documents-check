# 테이블 추출 시스템 - 병합된 셀 처리 개선

이 프로젝트는 구조도면의 테이블에서 데이터를 추출하는 시스템으로, 특히 **병합된 셀**을 정확히 처리하는 새로운 컴퓨터 비전 기반 방식을 제공합니다.

## 🚀 주요 기능

### 1. 두 가지 추출 방식 제공
- **Gemini AI 방식** (기존): AI 모델이 이미지를 분석하여 테이블 추출
- **컴퓨터 비전 방식** (신규): 수평선/수직선 검출로 정확한 병합 셀 처리

### 2. 병합된 셀 처리 개선
- 수평선을 검출하여 **평균 행 높이** 계산
- 각 셀의 높이를 행 높이로 나누어 **정확한 스팬(span)** 계산
- 병합된 셀의 내용을 해당하는 모든 행에 **정확히 복제**

## 📁 파일 구조

```
structural-documents-check/
├── main.py                    # 메인 실행 파일 (방식 선택 가능)
├── table_cv_extraction.py     # 컴퓨터 비전 기반 추출 함수
├── test_cv_extraction.py      # 테스트 및 비교 스크립트
├── img-split-calculation/     # 입력 이미지 폴더
└── debug_output/             # 디버그 이미지 출력 폴더
```

## 🛠 설치 요구사항

```bash
pip install opencv-python numpy pandas pillow google-generativeai
```

## 📖 사용법

### 1. 기본 실행 (방식 선택)

```bash
python main.py
```

실행하면 다음 옵션을 선택할 수 있습니다:
1. **Gemini AI 방식** (기존)
2. **컴퓨터 비전 방식** (병합셀 처리 개선)
3. **두 방식 모두** 실행하여 결과 비교

### 2. 테스트 및 디버깅

```bash
python test_cv_extraction.py
```

- 단일 이미지로 새로운 방식 테스트
- 디버그 이미지 생성으로 처리 과정 시각화
- 두 방식의 결과 비교

## 🔍 컴퓨터 비전 방식의 작동 원리

### 1단계: 수평선 검출
```python
# 이진화 후 수평선만 추출
horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
horiz_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, horiz_kernel)
```

### 2단계: 평균 행 높이 계산
```python
# 수평선들의 y 좌표 차이로 행 높이 계산
ys = sorted([cv2.boundingRect(cnt)[1] for cnt in contours])
row_height = np.median(np.diff(ys))
```

### 3단계: 셀 스팬 계산 및 데이터 복제
```python
# 셀 높이 ÷ 행 높이 = 스팬
span = max(1, int(round(h / row_height)))
start_row = max(0, int(round(y / row_height)))

# 스팬만큼 텍스트 복제
for i in range(span):
    df.iat[start_row + i, col_idx] = text
```

## 📊 결과 파일

실행 후 다음과 같은 CSV 파일이 생성됩니다:
- `table_extraction_results_ai_YYYYMMDD_HHMMSS.csv` (AI 방식)
- `table_extraction_results_cv_YYYYMMDD_HHMMSS.csv` (CV 방식)

## 🐛 디버깅

`test_cv_extraction.py` 실행 시 `debug_output/` 폴더에 생성되는 이미지들:

1. **01_original.png**: 원본 이미지
2. **02_horizontal_lines.png**: 검출된 수평선
3. **03_vertical_lines.png**: 검출된 수직선  
4. **04_detected_cells.png**: 검출된 셀 영역 (번호 표시)

## ⚡ 성능 비교

| 방식 | 장점 | 단점 |
|------|------|------|
| **Gemini AI** | 빠른 처리, 복잡한 레이아웃 처리 | 병합셀 처리 불정확, API 비용 |
| **컴퓨터 비전** | **정확한 병합셀 처리**, 구조적 분석 | 처리 시간 증가, 이미지 품질 의존적 |

## 🔧 고급 설정

### 병합 셀 감지 정확도 조정
`table_cv_extraction.py`에서 다음 파라미터를 조정할 수 있습니다:

```python
# 수평선 검출 커널 크기 (길수록 더 긴 선만 검출)
horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))

# 이상치 필터링 범위
filtered_heights = [h for h in row_heights if 0.3 * median_height <= h <= 3 * median_height]
```

### 멀티스레딩 조정
```python
# main.py에서 스레드 수 조정
process_calculations(api_key, max_workers=4)  # 기본값: 4
```

## 🚨 주의사항

1. **API 키 설정**: `GEMINI_API_KEY` 환경변수 설정 필요
2. **이미지 품질**: 선명한 이미지일수록 정확한 결과
3. **테이블 형식**: 격자형 테이블에 최적화됨
4. **폰트 크기**: 너무 작은 텍스트는 OCR 정확도 저하

## 📈 개선 사항

컴퓨터 비전 방식의 주요 개선점:

✅ **정확한 병합 셀 처리**: 5개 행이 병합된 셀도 정확히 5번 복제  
✅ **구조적 분석**: 실제 테이블 구조를 기반으로 데이터 배치  
✅ **디버깅 지원**: 시각적 디버깅으로 문제점 파악 가능  
✅ **재현 가능성**: 같은 이미지에 대해 일관된 결과  

## 🎯 사용 시나리오

### 병합된 셀이 많은 구조도면 테이블
- 기존 AI 방식: 병합 정보 누락
- **새로운 CV 방식**: 정확한 병합 처리로 완전한 데이터 추출

### 정확한 데이터 검증이 필요한 경우
- 두 방식 모두 실행하여 결과 비교
- 불일치 발견 시 CV 방식 결과 우선 검토