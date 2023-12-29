from ultralytics import YOLO
import cv2
import sys

# 명령행 인자로 전달된 이미지 파일 경로를 가져옵니다.
img_path = sys.argv[1]

# YOLO 모델을 초기화하고 미리 학습된 가중치를 로드합니다.
model = YOLO("Empire Building weights.pt")

# 이미지를 OpenCV를 사용하여 읽고 RGB 형식으로 변환합니다.
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# YOLO 모델을 사용하여 이미지에서 객체를 예측합니다.
# confidence threshold는 0.5로 설정되어 있습니다.
results = model.predict(img, conf=0.5)

# 예측 결과를 확인하고 해당하는 객체가 있는지 여부를 출력합니다.
for r in results:
    if r.boxes:
        print('True')  # 객체가 감지되었을 경우 True
    else:
        print('False')  # 객체가 감지되지 않았을 경우 False

# 실행방법 
# python hw2_1.py city1.jpg  OR  python hw2_1.py city2.jpg

