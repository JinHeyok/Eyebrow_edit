# pip install gdown
# pip install opencv-python
# pip install dlib
# pip install flask
# pip install boto3
 
from flask import Flask
from flask import request
from flask import jsonify
import dlib
import numpy as np
import cv2
import boto3


def s3_connection():
    try:
        # s3 클라이언트 생성
        s3 = boto3.client(
            service_name="s3",
            region_name="ap-northeast-2",
            aws_access_key_id="AKIAS5XD4RHI7ZXHKO7C",
            aws_secret_access_key="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",
        )
    except Exception as e:
        print(e)
    else:
        print("s3 bucket connected!") 
        return s3
        
s3 = s3_connection()

def hex_to_rgb(hex_value):
    hex_value = hex_value.lstrip('#')  # '#' 문자 제거
    hex_len = len(hex_value)
    
    if hex_len != 6:
        raise ValueError("유효한 6자리 Hex 값이어야 합니다.")
    
    # 10진수로 변환
    decimal_value = int(hex_value, 16)
    
    # RGB 값 추출
    red = (decimal_value >> 16) & 0xFF
    green = (decimal_value >> 8) & 0xFF
    blue = decimal_value & 0xFF
    
    return red, green, blue


app = Flask(__name__)
## 민무늬 눈썹
@app.route("/plain" , methods=["POST"])
def plain():
    imageName = request.json['imageName']
    
    # 얼굴 및 눈썹 검출을 위한 Dlib 초기화
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./images/shape_predictor_68_face_landmarks.dat")


    image = cv2.imread('./images/' + imageName)
    # Face detection using Haarcascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


    # Assuming there's only one face in the image
    x, y, w, h = faces[0]
    face_roi = image[y:y+h, x:x+w]


    # Forehead region selection (adjust the height as needed)
    forehead_roi = face_roi[:h//2, :]


    # Calculate the average color for the forehead region
    average_forehead_color = np.mean(np.mean(forehead_roi, axis=0), axis=0)
    average_forehead_color = average_forehead_color.astype(int)


    # Red, Green, Blue components
    avg_forehead_red, avg_forehead_green, avg_forehead_blue = average_forehead_color
    print(f"Forehead의 평균 RGB 값: ({avg_forehead_red}, {avg_forehead_green}, {avg_forehead_blue})")


    # 이미지 불러오기
    image_path = './images/' + imageName
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = detector(gray)
    for face in faces:
        # 얼굴 랜드마크 검출
        landmarks = predictor(gray, face)
        # 눈썹 영역 지정 (랜드마크의 좌표를 사용)
        eyebrow_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 27)])

        print(eyebrow_points)

        epsilon = 0.001 * cv2.arcLength(eyebrow_points, True)
        approx = cv2.approxPolyDP(eyebrow_points, epsilon, True)
        # 눈썹 영역을 감싸는 외곽선 안의 영역을 채워주기
        cv2.fillPoly(image, [approx], (int(avg_forehead_red)+30, int(avg_forehead_green)+30, int(avg_forehead_blue)+30))


        # 눈썹 부분을 추출하여 블러 처리
        y_min = min(eyebrow_points[:, 1])
        y_max = max(eyebrow_points[:, 1])
        x_min = min(eyebrow_points[:, 0])
        x_max = max(eyebrow_points[:, 0])

        # Ensure the coordinates are within the image boundaries
        y_min = max(0, y_min - 10)
        y_max = min(image.shape[0], y_max + 5)
        x_min = max(0, x_min)
        x_max = min(image.shape[1], x_max)

        eyebrow_roi = image[y_min:y_max, x_min:x_max]
        blurred_eyebrow_roi = cv2.GaussianBlur(eyebrow_roi, (25, 25),0)
        #blurred_eyebrow_roi = cv2.blur(eyebrow_roi, (11, 11),0)

        image[y_min:y_max, x_min:x_max] = blurred_eyebrow_roi

        # 결과를 표시하거나 파일로 저장합니다.
        plainImage = "plain" + imageName
        cv2.imwrite("./images/plain/" + plainImage, image)

        try:
            s3.upload_file("./images/plain/" + plainImage ,"blossom-cloud", "images/" + plainImage)
        except Exception as e:
            print(e)
        response = {"plainImage" : plainImage}
        return jsonify(response)

@app.route("/synthetic", methods=["POST"])
def synthetic():
    
    # 합성할 민무늬 이미지 이름
    imageName = request.json['imageName']
    # 합성할 눈썹 이미지 이름
    eyebrowImage = request.json["eyebrowImage"]
    # 눈썹 색상
    eyebrowColor = request.json['eyebrowColor']

    # HEX -> RGB 색상 바꾸기 
    hex_color = eyebrowColor
    rgb_color = hex_to_rgb(eyebrowColor)
    print(f"Hex: {hex_color}, RGB: {rgb_color}")
    
    # 얼굴 및 눈썹 검출을 위한 dlib 초기화
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./images/shape_predictor_68_face_landmarks.dat")

    # 얼굴 이미지와 눈썹 이미지 로드
    face_image = cv2.imread("./images/plain/" + imageName)
    # eyebrow_image = cv2.imread("./images/eyebrow.png", cv2.IMREAD_UNCHANGED)
    eyebrow_image = cv2.imread("./images/" + eyebrowImage, cv2.IMREAD_UNCHANGED)
    
    # 얼굴 색상 변경
    eyebrow_image[:, :, :3] = rgb_color

    # 얼굴 이미지를 그레이스케일로 변환
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    # 얼굴에서 눈썹 영역 찾기
    faces = detector(gray_face)
    for face in faces:
        landmarks = predictor(gray_face, face)

        # 눈썹 영역 추출 (랜드마크의 좌표를 사용)
        left_eyebrow_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 27)])

        # 눈썹 이미지 크기 조정
        eyebrow_height, eyebrow_width, _ = eyebrow_image.shape
        left_eyebrow_width = left_eyebrow_points[-1, 0] - left_eyebrow_points[0, 0] 
        scale_factor = left_eyebrow_width / eyebrow_width
        eyebrow_resized = cv2.resize(eyebrow_image, (int(eyebrow_width * scale_factor + 40), int(eyebrow_height * scale_factor)))

        # 눈썹 이미지를 얼굴 이미지에 합치기
        x_min = left_eyebrow_points[0, 0] - 20
        x_max = x_min + eyebrow_resized.shape[1]
        y_min = left_eyebrow_points[0, 1] - eyebrow_resized.shape[0] // 2
        y_max = y_min + eyebrow_resized.shape[0]

        # 눈썹 이미지를 합쳐진 얼굴 이미지에 적용
        for c in range(0, 3):
            face_image[y_min:y_max, x_min:x_max, c] = eyebrow_resized[:, :, c] * (eyebrow_resized[:, :, 3] / 255.0) + \
                                                      face_image[y_min:y_max, x_min:x_max, c] * (1.0 - eyebrow_resized[:, :, 3] / 255.0)
    syntheticImage = imageName.replace("plain" , "synthetic")
    cv2.imwrite("./images/synthetic/" + syntheticImage , face_image)
    try:
      s3.upload_file("./images/synthetic/" + syntheticImage ,"blossom-cloud", "images/" + syntheticImage)
    except Exception as e:
        print(e)

    response = {"syntheticImage" : syntheticImage}
    return jsonify(response)   

@app.route("/hello", methods=["GET"])
def hello():
    return "hello World"

if __name__ == '__main__':
    app.run()