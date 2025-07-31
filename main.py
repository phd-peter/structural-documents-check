import os
import glob
import csv
import importlib.util
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# gemini-ocr.py에서 필요한 함수들 import (하이픈 때문에 동적 import 사용)
spec = importlib.util.spec_from_file_location("gemini_ocr", "gemini-ocr.py")
gemini_ocr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gemini_ocr)

# 필요한 함수들을 변수에 할당
extract_front_info_gemini = gemini_ocr.extract_front_info_gemini
convert_date_format = gemini_ocr.convert_date_format

def extract_table_data_gemini(api_key, image_path: str):
    """표 형식 이미지에서 데이터를 추출하는 함수"""
    from google import genai
    from google.genai import types
    import json
    import re
    
    client = genai.Client(api_key=api_key)
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            (
                """
                이 이미지에는 표 형식의 데이터가 포함되어 있습니다. 당신의 임무는 이 표를 정확하게 인식하고 구조화된 데이터로 변환하는 것입니다.

                ## 분석 지침:
                1. **표 구조 파악**: 행(rows)과 열(columns)의 경계를 명확히 구분하세요
                2. **헤더 인식**: 첫 번째 행이 헤더인지 확인하고, 각 열의 제목을 정확히 추출하세요
                3. **셀 내용 추출**: 각 셀의 텍스트를 정확하게 읽고, 숫자는 숫자로, 텍스트는 텍스트로 구분하세요
                4. **병합된 셀 처리**: 병합된 셀이 있다면 해당 내용을 적절한 위치에 배치하세요
                5. **공백 처리**: 빈 셀은 빈 문자열("")로 표시하세요

                ## 특별 주의사항:
                - 열과 행이 선으로 분리되어 있지 않더라도, 데이터의 정렬과 간격을 보고 구조를 파악하세요
                - 숫자 데이터에서 쉼표(,)나 원화 표시(₩) 등은 제거하고 순수 숫자만 추출하세요
                - 날짜 형식은 원본 그대로 유지하세요
                - 텍스트에서 불필요한 공백은 제거하되, 의미 있는 내용은 보존하세요

                ## 출력 형식:
                다음 JSON 형식으로 정확히 반환해주세요:
                {
                    "headers": ["열1제목", "열2제목", "열3제목", ...],
                    "rows": [
                        ["행1열1값", "행1열2값", "행1열3값", ...],
                        ["행2열1값", "행2열2값", "행2열3값", ...],
                        ...
                    ]
                }

                만약 헤더가 없는 표라면 headers는 ["Column1", "Column2", "Column3", ...] 형태로 만들어주세요.
                """
            ),
        ],
    )
    
    raw = response.text.strip()
    # 코드블록 백틱이 있을 경우 제거
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    
    return json.loads(raw)

def process_single_calculation(api_key, image_path, index):
    """단일 계산서 처리 함수 (멀티스레딩용)"""
    print(f"{index}번째 계산서 처리 시작: {os.path.basename(image_path)}")
    try:
        table_data = extract_table_data_gemini(api_key, image_path)
        print(f"{index}번째 계산서 완료: {os.path.basename(image_path)}")
        print("추출된 표 데이터:", table_data)
        
        # 표 데이터를 CSV 형태로 변환
        result = []
        headers = table_data.get('headers', [])
        rows = table_data.get('rows', [])
        
        # 헤더를 첫 번째 행으로 추가 (파일명 포함)
        if headers:
            result.append([os.path.basename(image_path)] + headers)
        
        # 각 데이터 행을 추가 (파일명은 첫 번째 행에만)
        for i, row in enumerate(rows):
            if i == 0:
                result.append([''] + row)  # 첫 번째 데이터 행
            else:
                result.append([''] + row)  # 나머지 행들
        
        return result
        
    except Exception as e:
        print(f"{index}번째 계산서 오류: {e}")
        return [[os.path.basename(image_path), 'ERROR', str(e)]]

def process_calculations(api_key, max_workers=4):
    """img-calculation 폴더의 표 형식 이미지들을 동시에 처리하여 정보를 추출하고 CSV로 저장"""
    
    # img-calculation 폴더 확인
    calculation_folder = "img-calculation"
    if not os.path.exists(calculation_folder):
        print(f"{calculation_folder} 폴더가 없습니다.")
        return
    
    # 이미지 파일들 찾기
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(calculation_folder, ext)))
    image_files.sort()
    
    if not image_files:
        print(f"{calculation_folder} 폴더에 이미지 파일이 없습니다.")
        return
    
    print(f"총 {len(image_files)}개 표 이미지를 {max_workers}개 스레드로 동시 처리합니다.")
    
    results = [None] * len(image_files)  # 순서 보장을 위한 리스트
    
    # ThreadPoolExecutor를 사용한 동시 처리
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 작업 제출
        future_to_index = {
            executor.submit(process_single_calculation, api_key, image_path, i+1): i 
            for i, image_path in enumerate(image_files)
        }
        
        # 완료된 작업들 수집
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as e:
                print(f"작업 실패: {e}")
                results[index] = [[os.path.basename(image_files[index]), 'ERROR', str(e)]]
    
    # CSV 저장 - 타임스탬프로 고유한 파일명 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'table_extraction_results_{timestamp}.csv'
    
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            # 모든 표 데이터를 하나의 CSV로 통합
            first_file = True
            for table_result in results:
                if table_result:
                    for row_idx, row in enumerate(table_result):
                        if first_file and row_idx == 0:
                            # 첫 번째 파일의 헤더 행을 CSV 헤더로 사용
                            writer.writerow(['source_file'] + row[1:])  # 파일명 열 + 원본 헤더
                            first_file = False
                        elif row_idx == 0:
                            # 다른 파일들의 헤더는 건너뜀 (중복 방지)
                            continue
                        else:
                            # 데이터 행들을 저장
                            source_file = os.path.basename(image_files[results.index(table_result)])
                            writer.writerow([source_file] + row[1:])
        
        print(f"{csv_filename}에 저장완료!")
        print(f"총 {len([r for r in results if r])}개 표가 처리되었습니다.")
            
    except Exception as e:
        print(f"❌ CSV 저장 중 오류 발생: {e}")
        print("\n📋 추출된 표 데이터:")
        for i, table_result in enumerate(results):
            if table_result:
                print(f"\n=== {os.path.basename(image_files[i])} ===")
                for row in table_result:
                    print(','.join(str(item) for item in row))

if __name__ == "__main__":
    # API 키 설정 (환경변수에서 가져오거나 직접 입력)
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        api_key = input("Gemini API 키를 입력하세요: ")
    
    if api_key:
        # max_workers 파라미터로 동시 처리할 스레드 수 조절 (기본값: 4)
        process_calculations(api_key, max_workers=4)
    else:
        print("API 키가 필요합니다.")
