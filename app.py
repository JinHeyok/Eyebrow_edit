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
import threading
import subprocess
import os 


def s3_connection():
    try:
        # s3 클라이언트 생성
        s3 = boto3.client(
            service_name="s3",
            region_name="",
            aws_access_key_id="",
            aws_secret_access_key="",
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
lock = threading.Lock()
## 민무늬 눈썹
@app.route("/plain" , methods=["POST"])
def plain():
    with lock : 
        originalImageName = request.json['imageName']
        faceAnalysisBatPath = "./face_analysis/face_analysis.bat"
        
        # 파일이름으로 배치파일 생성
        with open(faceAnalysisBatPath , "w") as file :
            # 배치파일있느 경로 설정 
            file.write("cd /face_analysis" + "\n")
            file.write("face_analysis.exe " + originalImageName + "\n")
            file.write("exit")
            
        # 배치 파일 실행 
        subprocess.call([f'.\\face_analysis\\face_analysis.bat'])
        
        jsonFile = "info_" + originalImageName  # 눈썹을 지울 때 생성될 때 생기는 JSON 파일
        eraseFile = "erase_" + originalImageName # 눈썹이 지워진 최종 사진
        resizeFile= "resize_" + originalImageName # 눈썹을 지울 때 생성되는 Reszing 파일
        
        # TODO 경로 수정 필요
        # image = cv2.imread('./face_analysis/' + eraseFile)

        # 결과를 표시하거나 파일로 저장합니다.
        # cv2.imwrite("./images/plain/" + eraseFile, image)
        # cv2.imwrite("./images/plain/" + eraseFile, image)

        # 눈썹 지워진 사진 S3 업로드
        try:
            s3.upload_file("./face_analysis/" + eraseFile ,"butket name", "images/" + eraseFile)
        except Exception as e:
            print(e)
            
        os.remove(r"./face_analysis/" + originalImageName)    
        os.remove(r"./face_analysis/" + eraseFile)    
        os.remove(r"./face_analysis/" + resizeFile)    
        os.remove(r"./face_analysis/" + jsonFile.split(".")[0] + ".json")    
            
        
        response = {"plainImage" : eraseFile}
        return jsonify(response)

@app.route("/synthetic", methods=["POST"])
def synthetic():
    with lock : 
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
        predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

        # 얼굴 이미지와 눈썹 이미지 로드
        face_image = cv2.imread("./images/plain/" + imageName)
        # eyebrow_image = cv2.imread("./images/eyebrow.png", cv2.IMREAD_UNCHANGED)
        eyebrow_image = cv2.imread("./images/eyebrow/" + eyebrowImage, cv2.IMREAD_UNCHANGED)
        
        # 얼굴 색상 변경
        # eyebrow_image[:, :, :3] = rgb_color RGB가 반대입니다.
        eyebrow_image[:, :, :3] = (rgb_color[2],rgb_color[1],rgb_color[0])
        
        # 얼굴 이미지를 그레이스케일로 변환
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        # 얼굴에서 눈썹 영역 찾기
        faces = detector(gray_face)
        for face in faces:
            landmarks = predictor(gray_face, face)

            # 눈썹 영역 추출 (랜드마크의 좌표를 사용)
            left_eyebrow_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 27)])

            # 눈썹 이미지 크기 조정!
            eyebrow_height, eyebrow_width, _ = eyebrow_image.shape
            left_eyebrow_width = left_eyebrow_points[-1, 0] - left_eyebrow_points[0, 0] 
            scale_factor = left_eyebrow_width / eyebrow_width
            eyebrow_resized = cv2.resize(eyebrow_image, (int(eyebrow_width * scale_factor ), int(eyebrow_height * scale_factor)))

           # 눈썹 이미지를 얼굴 이미지에 합치기
            x_min = left_eyebrow_points[0, 0] 
            x_max = x_min + eyebrow_resized.shape[1]
            y_min = left_eyebrow_points[0, 1] - eyebrow_resized.shape[0]
            y_max = y_min + eyebrow_resized.shape[0]

            # 눈썹 이미지를 합쳐진 얼굴 이미지에 적용
            for c in range(0, 3):
                face_image[y_min:y_max, x_min:x_max, c] = eyebrow_resized[:, :, c] * (eyebrow_resized[:, :, 3] / 255.0) + \
                                                            face_image[y_min:y_max, x_min:x_max, c] * (1.0 - eyebrow_resized[:, :, 3] / 255.0)
        syntheticImage = "synthetic" + imageName
        cv2.imwrite("./images/synthetic/" + syntheticImage , face_image)
        try:
            s3.upload_file("./images/synthetic/" + syntheticImage ,"blossom-cloud", "images/" + syntheticImage)
        except Exception as e:
            print(e)

        # 이미지 삭제
        os.remove(r"./images/synthetic/" + syntheticImage)
        os.remove(r"./images/plain/" + imageName)
        # 이미지 삭제
        
        response = {"syntheticImage" : syntheticImage}
        return jsonify(response)   

app.config['THREADED'] = True
if __name__ == '__main__':
    app.run(threaded=True, use_reloader=True)
    