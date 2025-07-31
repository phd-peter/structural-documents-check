import cv2
import matplotlib.pyplot as plt

# 원본 이미지 로드
img = cv2.imread('/Users/peter/Projects/structural-documents-check/img-split-drawing/center_img_draw.png')

# 1. 그레이스케일 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 이진화 (OTSU 방법)
_, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 3. 수평선만 남기기
horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
horiz_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, horiz_kernel)

# 결과 이미지들을 한번에 표시
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 원본 이미지 (BGR을 RGB로 변환)
axes[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0,0].set_title('Original Image')
axes[0,0].axis('off')

# 그레이스케일 이미지
axes[0,1].imshow(gray, cmap='gray')
axes[0,1].set_title('Grayscale')
axes[0,1].axis('off')

# 이진화 이미지
axes[1,0].imshow(bin_img, cmap='gray')
axes[1,0].set_title('Binary (OTSU)')
axes[1,0].axis('off')

# 수평선 추출 결과
axes[1,1].imshow(horiz_lines, cmap='gray')
axes[1,1].set_title('Horizontal Lines')
axes[1,1].axis('off')

plt.tight_layout()
plt.show()

# 개별 이미지도 저장
cv2.imwrite('result_gray.png', gray)
cv2.imwrite('result_binary.png', bin_img)
cv2.imwrite('result_horiz_lines.png', horiz_lines)

print("변환된 이미지들이 표시되었습니다.")
print("개별 파일로도 저장되었습니다: result_gray.png, result_binary.png, result_horiz_lines.png")