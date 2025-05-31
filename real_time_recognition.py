import cv2
import dlib
import numpy as np
import pickle

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
    faces = detector(rgb, 1)

    for face in faces:
        # 提取实时人脸特征
        shape = shape_predictor(rgb, face)
        descriptor = face_encoder.compute_face_descriptor(rgb, shape)
        current_feature = np.array(descriptor)

        # 计算与已知人脸的欧氏距离
        min_distance = float("inf")
        identity = "Unknown"
        for name, saved_feature in name_features.items():
            distance = np.linalg.norm(current_feature - saved_feature)
            if distance < min_distance and distance < RECOGNITION_THRESHOLD:
                min_distance = distance
                identity = name

        # 绘制识别结果
        cv2.rectangle(frame, (face.left(), face.top()),
                      (face.right(), face.bottom()), (0, 255, 0), 2)
        cv2.putText(frame, f"{identity}", (face.left() + 5, face.top() - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


# 启动摄像头
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = recognize_face(frame)
    cv2.imshow("Real-time Face Recognition", frame)

    if cv2.waitKey(1) == 27:  # ESC键退出
        break

cap.release()
cv2.destroyAllWindows()