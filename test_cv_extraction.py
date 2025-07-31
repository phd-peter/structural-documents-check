#!/usr/bin/env python3
"""
컴퓨터 비전 기반 테이블 추출 방식 테스트 스크립트
"""

import os
from table_cv_extraction import extract_table_with_cv, save_debug_images
import cv2

def test_single_image(api_key, image_path):
    """단일 이미지로 컴퓨터 비전 방식 테스트"""
    print(f"=== 테스트 이미지: {image_path} ===")
    
    if not os.path.exists(image_path):
        print(f"이미지 파일이 존재하지 않습니다: {image_path}")
        return
    
    try:
        # 이미지 로드하여 디버그 이미지도 생성
        img = cv2.imread(image_path)
        if img is None:
            print("이미지를 로드할 수 없습니다.")
            return
        
        print("1단계: 이미지 전처리 및 디버그 이미지 생성...")
        
        # 디버그 이미지 생성을 위해 내부 함수들 호출
        from table_cv_extraction import detect_horizontal_lines, calculate_row_height, detect_vertical_lines, detect_table_cells
        
        horiz_lines, bin_img, gray = detect_horizontal_lines(img)
        row_height = calculate_row_height(horiz_lines)
        vert_lines = detect_vertical_lines(bin_img)
        cell_boxes = detect_table_cells(img, horiz_lines, vert_lines)
        
        # 디버그 이미지 저장
        save_debug_images(img, horiz_lines, vert_lines, cell_boxes)
        
        print("2단계: 테이블 추출 실행...")
        result = extract_table_with_cv(api_key, image_path)
        
        print("\n=== 추출 결과 ===")
        print(f"헤더: {result['headers']}")
        print(f"데이터 행 개수: {len(result['rows'])}")
        
        for i, row in enumerate(result['rows']):
            print(f"행 {i+1}: {row}")
        
        print(f"\n디버그 이미지가 'debug_output' 폴더에 저장되었습니다.")
        print("- 01_original.png: 원본 이미지")
        print("- 02_horizontal_lines.png: 검출된 수평선")
        print("- 03_vertical_lines.png: 검출된 수직선")
        print("- 04_detected_cells.png: 검출된 셀 영역")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

def compare_methods(api_key, image_path):
    """두 방식의 결과를 비교"""
    print(f"=== 두 방식 비교: {image_path} ===")
    
    if not os.path.exists(image_path):
        print(f"이미지 파일이 존재하지 않습니다: {image_path}")
        return
    
    # 기존 Gemini AI 방식
    try:
        from main import extract_table_data_gemini
        print("\n1. Gemini AI 방식:")
        ai_result = extract_table_data_gemini(api_key, image_path)
        print(f"헤더: {ai_result['headers']}")
        print(f"데이터 행 개수: {len(ai_result['rows'])}")
        for i, row in enumerate(ai_result['rows']):
            print(f"  행 {i+1}: {row}")
    except Exception as e:
        print(f"Gemini AI 방식 오류: {e}")
        ai_result = None
    
    # 새로운 컴퓨터 비전 방식
    try:
        print("\n2. 컴퓨터 비전 방식:")
        cv_result = extract_table_with_cv(api_key, image_path)
        print(f"헤더: {cv_result['headers']}")
        print(f"데이터 행 개수: {len(cv_result['rows'])}")
        for i, row in enumerate(cv_result['rows']):
            print(f"  행 {i+1}: {row}")
    except Exception as e:
        print(f"컴퓨터 비전 방식 오류: {e}")
        cv_result = None
    
    # 결과 비교
    if ai_result and cv_result:
        print("\n=== 결과 비교 ===")
        print(f"AI 방식 데이터 행 수: {len(ai_result['rows'])}")
        print(f"CV 방식 데이터 행 수: {len(cv_result['rows'])}")
        
        if len(ai_result['rows']) != len(cv_result['rows']):
            print("⚠️  두 방식의 데이터 행 개수가 다릅니다!")
            print("   컴퓨터 비전 방식이 병합된 셀을 더 정확히 처리했을 가능성이 있습니다.")

if __name__ == "__main__":
    # API 키 설정
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        api_key = input("Gemini API 키를 입력하세요: ")
    
    if not api_key:
        print("API 키가 필요합니다.")
        exit(1)
    
    # 테스트할 이미지 찾기
    test_images = []
    
    # img-split-calculation 폴더에서 이미지 찾기
    calc_folder = "img-split-calculation"
    if os.path.exists(calc_folder):
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            import glob
            test_images.extend(glob.glob(os.path.join(calc_folder, ext)))
    
    if not test_images:
        print("테스트할 이미지를 찾을 수 없습니다.")
        print("img-split-calculation 폴더에 이미지가 있는지 확인하세요.")
        exit(1)
    
    # 첫 번째 이미지로 테스트
    test_image = test_images[0]
    print(f"테스트 이미지: {test_image}")
    
    # 테스트 모드 선택
    print("\n=== 테스트 모드 선택 ===")
    print("1. 컴퓨터 비전 방식만 테스트 (디버그 이미지 포함)")
    print("2. 두 방식 비교 테스트")
    
    while True:
        choice = input("선택하세요 (1/2): ").strip()
        if choice in ['1', '2']:
            break
        print("1 또는 2를 입력하세요.")
    
    if choice == '1':
        test_single_image(api_key, test_image)
    else:
        compare_methods(api_key, test_image)