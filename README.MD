## 눈썹 합성
민무늬 이미지 합성


    Request
    {
        "imageName" : "test.jpg" // 얼굴 이미지
    }
---

    Response
    {
        "plainImage": "plaintest.jpg"
    }

---
민무늬 눈썹 이미지로 눈썹 합성

    Requset
    {
        "imageName" : "plaintest.jpg", //민무늬 이미지 
        "eyebrowImage" : "eyebrow.png", // 눈썹 이미지 
        "eyebrowColor" : "#C4FF00" // 변경할 색상
    }
--- 
    Response
    {
        "syntheticImage": "synthetictest.jpg"
    }