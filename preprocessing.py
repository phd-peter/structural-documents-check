from pdf2image import convert_from_path

# Poppler 경로 (다운로드 후 압축 해제한 경로로 수정하세요)
poppler_path = r'C:\poppler\Library\bin'

drawing_pdf_path = r'C:\Users\Alpha\Projects\structural-documents-check\drawing-101.pdf'
calculation_pdf_path = r'C:\Users\Alpha\Projects\structural-documents-check\calculation-101.pdf'
drawing_output_folder = r'C:\Users\Alpha\Projects\structural-documents-check\img-drawing'
calculation_output_folder = r'C:\Users\Alpha\Projects\structural-documents-check\img-calculation'
dpi = 200

## drawing
# 101동: 330~332 페이지 (인덱스는 0부터 시작)
page_numbers = [329, 330, 331]

images = convert_from_path(drawing_pdf_path, dpi=dpi, first_page=min(page_numbers)+1, last_page=max(page_numbers)+1, poppler_path=poppler_path)

# 저장
for idx, page_num in enumerate(page_numbers):
    images[idx].save(f"{drawing_output_folder}\\drawing_{idx}.png", 'PNG')


## calculations
# 101동: 73-75 페이지
page_numbers = [72, 73, 74]

images = convert_from_path(calculation_pdf_path, dpi=dpi, first_page=min(page_numbers)+1, last_page=max(page_numbers)+1, poppler_path=poppler_path)

# 저장
for idx, page_num in enumerate(page_numbers):
    images[idx].save(f"{calculation_output_folder}\\calculation_{idx}.png", 'PNG')