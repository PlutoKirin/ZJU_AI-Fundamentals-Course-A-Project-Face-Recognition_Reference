import cv2
import dlib
import numpy as np
import pickle
import os

# 加载模型和特征数据
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
with open("face_features.pkl", "rb") as f:
    name_features = pickle.load(f)

# 设定阈值（可调整）
RECOGNITION_THRESHOLD = 0.45


def recognize_face(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(rgb, 1)
    faces1 = detector(gray, 0)
    for k,d in enumerate(faces):
        height = (d.bottom() - d.top())
        width = (d.right() - d.left())
        hh = int(height / 16)
        ww = int(width / 16)
        x1,y1=tuple([d.left() - ww*2, d.top() - hh*2])
        x2,y2=tuple([d.right() + ww*2, d.bottom() + hh*2])
        face_img=cv2.resize(frame[y1:y2,x1:x2],(256,256),interpolation=cv2.INTER_AREA)
    for face in faces:
        # 提取实时人脸特征
        shape = shape_predictor(rgb, face)
        descriptor = face_encoder.compute_face_descriptor(rgb, shape)
        current_feature = np.array(descriptor)

        # 计算与已知人脸的欧氏距离
        min_distance = float("inf")
        identity = "Unknown"
        confidence = 0.0  # 初始化置信度

        for name, saved_feature in name_features.items():
            distance = np.linalg.norm(current_feature - saved_feature)
            if distance < min_distance:
                min_distance = distance
                identity = name
                # 将距离转换为置信度百分比（距离越小置信度越高）
                confidence = max(0, 1 - distance / RECOGNITION_THRESHOLD) * 100

        # 如果最小距离仍超过阈值，标记为Unknown
        if min_distance > RECOGNITION_THRESHOLD:
            identity = "Unknown"
            confidence = 0.0

        # 绘制识别结果和置信度
     #   cv2.rectangle(frame, (face.left(), face.top()),
                  #    (face.right(), face.bottom()), (0, 255, 0), 2)

        # 显示识别结果（名称+置信度）

        text = f"{identity} ({confidence:.1f}%)"
       # cv2.putText(face_img, text, (face.left() + 5, face.top() - 15),
                 #  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(face_img, text, (0, 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return face_img


# 启动摄像头
image_dir = "demo"  # 替换为你的图片文件夹路径

# 遍历目录下所有图片
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)

    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"无法读取图片: {image_path}")
            continue
        # 调用人脸识别函数
        result_frame = recognize_face(frame)

        # 显示结果
        cv2.imshow(f"Face Recognition: {image_name}", result_frame)
        cv2.waitKey(0)  # 按任意键继续下一张

cv2.destroyAllWindows()