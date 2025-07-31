import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
from google import genai
from google.genai import types
import json
import re

def detect_horizontal_lines(img):
    """수평선을 검출하여 행 경계를 찾는 함수"""
    # 그레이스케일 변환
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 이진화 (OTSU 방법)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 수평선만 추출하기 위한 커널 (길이 40, 높이 1)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horiz_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, horiz_kernel)
    
    return horiz_lines, bin_img, gray

def calculate_row_height(horiz_lines):
    """수평선으로부터 평균 행 높이를 계산하는 함수"""
    contours, _ = cv2.findContours(horiz_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 수평선의 y 좌표들 수집
    ys = []
    for cnt in contours:
        _, y, _, h = cv2.boundingRect(cnt)
        ys.append(y)
    
    if len(ys) < 2:
        print("Warning: 충분한 수평선을 찾을 수 없습니다. 기본값을 사용합니다.")
        return 50  # 기본값
    
    ys = sorted(ys)
    # 인접한 y 좌표 간의 차이 계산
    row_heights = np.diff(ys)
    
    # 이상치 제거 (너무 작거나 큰 값들)
    median_height = np.median(row_heights)
    filtered_heights = [h for h in row_heights if 0.3 * median_height <= h <= 3 * median_height]
    
    if filtered_heights:
        avg_row_height = np.median(filtered_heights)
    else:
        avg_row_height = median_height
    
    print(f"평균 행 높이: {avg_row_height:.1f}px")
    return avg_row_height

def detect_vertical_lines(bin_img):
    """수직선을 검출하여 열 경계를 찾는 함수"""
    # 수직선 추출을 위한 커널 (길이 1, 높이 20)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    vert_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, vert_kernel)
    
    return vert_lines

def detect_table_cells(img, horiz_lines, vert_lines):
    """수평선과 수직선을 결합하여 셀 영역을 검출하는 함수"""
    # 수평선과 수직선 결합
    table_mask = cv2.add(horiz_lines, vert_lines)
    
    # 셀 영역 검출을 위해 테이블 마스크를 이용하여 그리드 생성
    joints = cv2.bitwise_and(horiz_lines, vert_lines)
    
    # 컨투어 검출로 셀 영역들 찾기
    contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cell_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 너무 작은 영역은 제외
        if w > 20 and h > 10:
            cell_boxes.append((x, y, w, h))
    
    # x 좌표로 정렬하여 열 순서대로 배치
    cell_boxes.sort(key=lambda box: (box[1], box[0]))  # y좌표 우선, x좌표 차순
    
    return cell_boxes

def calculate_cell_span(cell_box, row_height):
    """셀의 높이로부터 스팬(몇 개 행을 차지하는지)을 계산하는 함수"""
    x, y, w, h = cell_box
    span = max(1, int(round(h / row_height)))
    start_row = max(0, int(round(y / row_height)))
    
    return span, start_row

def extract_text_from_cell(img, cell_box, api_key):
    """개별 셀에서 OCR로 텍스트를 추출하는 함수"""
    x, y, w, h = cell_box
    
    # 셀 이미지 추출 (여백 추가)
    padding = 2
    cell_img = img[max(0, y-padding):min(img.shape[0], y+h+padding), 
                  max(0, x-padding):min(img.shape[1], x+w+padding)]
    
    if cell_img.size == 0:
        return ""
    
    try:
        # PIL Image로 변환
        if len(cell_img.shape) == 3:
            cell_pil = Image.fromarray(cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB))
        else:
            cell_pil = Image.fromarray(cell_img)
        
        # 임시 파일로 저장
        temp_path = "temp_cell.png"
        cell_pil.save(temp_path)
        
        # Gemini API로 OCR 수행
        client = genai.Client(api_key=api_key)
        with open(temp_path, "rb") as f:
            image_bytes = f.read()

        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                "이 이미지에서 텍스트를 정확히 추출해주세요. 공백은 제거하고 의미있는 내용만 반환하세요. 빈 이미지라면 빈 문자열을 반환하세요."
            ],
        )
        
        text = response.text.strip()
        
        # 임시 파일 삭제
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return text
        
    except Exception as e:
        print(f"OCR 오류: {e}")
        return ""

def extract_table_with_cv(api_key, image_path):
    """컴퓨터 비전 기반으로 테이블을 추출하는 메인 함수"""
    print(f"컴퓨터 비전 방식으로 테이블 추출 시작: {image_path}")
    
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    
    # 1단계: 수평선 검출
    horiz_lines, bin_img, gray = detect_horizontal_lines(img)
    
    # 2단계: 평균 행 높이 계산
    row_height = calculate_row_height(horiz_lines)
    
    # 3단계: 수직선 검출
    vert_lines = detect_vertical_lines(bin_img)
    
    # 4단계: 셀 영역 검출
    cell_boxes = detect_table_cells(img, horiz_lines, vert_lines)
    print(f"검출된 셀 개수: {len(cell_boxes)}")
    
    # 5단계: 테이블 구조 분석
    # 대략적인 열 개수 추정 (x 좌표 기준으로 그룹핑)
    x_coords = sorted(list(set([box[0] for box in cell_boxes])))
    num_cols = len(x_coords)
    
    # 대략적인 행 개수 추정
    max_y = max([box[1] + box[3] for box in cell_boxes])
    num_rows = int(max_y / row_height) + 1
    
    print(f"예상 테이블 크기: {num_rows}행 x {num_cols}열")
    
    # 6단계: DataFrame 초기화
    df = pd.DataFrame([[None for _ in range(num_cols)] for _ in range(num_rows)])
    
    # 7단계: 각 셀에서 텍스트 추출 및 스팬 처리
    for i, cell_box in enumerate(cell_boxes):
        try:
            # 텍스트 추출
            text = extract_text_from_cell(img, cell_box, api_key)
            
            if text:  # 빈 텍스트가 아닌 경우만 처리
                # 스팬 계산
                span, start_row = calculate_cell_span(cell_box, row_height)
                
                # 열 인덱스 추정 (x 좌표 기준)
                col_idx = 0
                for j, x_coord in enumerate(x_coords):
                    if abs(cell_box[0] - x_coord) < 20:  # 20px 오차 허용
                        col_idx = j
                        break
                
                # DataFrame에 텍스트 할당 (스팬만큼 복제)
                for row_offset in range(span):
                    target_row = start_row + row_offset
                    if 0 <= target_row < num_rows and 0 <= col_idx < num_cols:
                        df.iat[target_row, col_idx] = text
                
                print(f"셀 {i+1}: '{text}' -> 행{start_row}~{start_row+span-1}, 열{col_idx}")
        
        except Exception as e:
            print(f"셀 {i+1} 처리 중 오류: {e}")
            continue
    
    # 8단계: 결과 정리
    # 빈 행 제거
    df_cleaned = df.dropna(how='all').reset_index(drop=True)
    
    # 헤더와 데이터 분리
    if len(df_cleaned) > 0:
        headers = df_cleaned.iloc[0].tolist()
        data_rows = df_cleaned.iloc[1:].values.tolist()
        
        # None을 빈 문자열로 변경
        headers = [str(h) if h is not None else '' for h in headers]
        data_rows = [[str(cell) if cell is not None else '' for cell in row] for row in data_rows]
        
        return {
            'headers': headers,
            'rows': data_rows
        }
    else:
        return {
            'headers': [],
            'rows': []
        }

def save_debug_images(img, horiz_lines, vert_lines, cell_boxes, output_dir="debug_output"):
    """디버깅용 이미지 저장 함수"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 원본 이미지
    cv2.imwrite(f"{output_dir}/01_original.png", img)
    
    # 수평선
    cv2.imwrite(f"{output_dir}/02_horizontal_lines.png", horiz_lines)
    
    # 수직선
    cv2.imwrite(f"{output_dir}/03_vertical_lines.png", vert_lines)
    
    # 셀 박스 그리기
    img_with_boxes = img.copy()
    for i, (x, y, w, h) in enumerate(cell_boxes):
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_with_boxes, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.imwrite(f"{output_dir}/04_detected_cells.png", img_with_boxes)
    print(f"디버그 이미지들이 {output_dir} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    # 테스트용 코드
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        api_key = input("Gemini API 키를 입력하세요: ")
    
    test_image = "img-split-calculation/center_img.png"  # 테스트할 이미지 경로
    if os.path.exists(test_image):
        try:
            result = extract_table_with_cv(api_key, test_image)
            print("\n=== 추출 결과 ===")
            print("Headers:", result['headers'])
            print("Rows:")
            for i, row in enumerate(result['rows']):
                print(f"  Row {i+1}: {row}")
        except Exception as e:
            print(f"오류: {e}")
    else:
        print(f"테스트 이미지를 찾을 수 없습니다: {test_image}")