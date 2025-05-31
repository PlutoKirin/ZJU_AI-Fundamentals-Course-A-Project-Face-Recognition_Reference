import cv2
import dlib
import numpy as np
import gradio as gr
import pickle
import pandas as pd
from datetime import datetime
import os
import time
from scipy.spatial import distance as dist

# === 配置 ===
ATTENDANCE_FOLDER = "attendance_results"  # 签到结果文件夹
os.makedirs(ATTENDANCE_FOLDER, exist_ok=True)  # 自动创建文件夹

# 模型路径（与脚本同目录）
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FACE_ENCODER_PATH = "dlib_face_recognition_resnet_model_v1.dat"

# 评估图片路径
EVALUATION_FOLDER = "evaluation"
CONFUSION_MATRIX_PATH = os.path.join(EVALUATION_FOLDER, "confusion_matrix.png")
METRICS_PATH = os.path.join(EVALUATION_FOLDER, "metrics.png")


# 检查评估图片是否存在
def check_evaluation_images():
    """检查评估图片是否存在，不存在则返回None"""
    images = {}
    for name, path in [("confusion_matrix", CONFUSION_MATRIX_PATH),
                       ("metrics", METRICS_PATH)]:
        if os.path.exists(path) and os.path.isfile(path):
            images[name] = path
        else:
            images[name] = None
            print(f"警告：找不到评估图片 - {path}")
    return images


# 加载模型和特征数据
try:
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    face_encoder = dlib.face_recognition_model_v1(FACE_ENCODER_PATH)
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")
    raise

try:
    with open("face_features.pkl", "rb") as f:
        name_features = pickle.load(f)
    print(f"成功加载 {len(name_features)} 个人脸特征")
except (FileNotFoundError, pickle.UnpicklingError) as e:
    name_features = {}
    print(f"警告：无法加载人脸特征数据库 - {str(e)}")
    print("所有人脸将被识别为'Unknown'")

# 设定阈值（可调整）
RECOGNITION_THRESHOLD = 0.45

# === 签到系统状态 ===
signed_names = set()  # 已签到人员集合
attendance_records = []  # 签到记录
is_signing_active = False  # 签到系统是否激活


# === 工具函数 ===
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[9])
    B = np.linalg.norm(mouth[4] - mouth[7])
    C = np.linalg.norm(mouth[0] - mouth[6])
    return (A + B) / (2.0 * C)


def nod_aspect_ratio(size, pre_point, now_point):
    return abs(float((pre_point[1] - now_point[1]) / (size[0] / 2)))


def shake_aspect_ratio(size, pre_point, now_point):
    return abs(float((pre_point[0] - now_point[0]) / (size[1] / 2)))


def reset_attendance():
    """重置签到状态"""
    global signed_names, attendance_records, is_signing_active
    signed_names = set()
    attendance_records = []
    is_signing_active = False
    return "签到状态已重置", ""


def save_attendance_to_excel():
    """保存签到记录到Excel"""
    if not attendance_records:
        return "没有签到记录可保存"

    df = pd.DataFrame(attendance_records, columns=["姓名", "签到时间"])
    filename = f"签到记录_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    path = os.path.join(ATTENDANCE_FOLDER, filename)

    try:
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name="签到表")
        return f"签到结果已保存到：{path}"
    except Exception as e:
        return f"保存失败: {str(e)}"


def get_signed_names_text():
    """获取已签到人员的文本列表"""
    if not signed_names:
        return "暂无签到记录"
    return "\n".join([f"{i + 1}. {name}" for i, name in enumerate(sorted(signed_names))])


# === 人脸识别函数 ===
def recognize_face(frame, threshold=RECOGNITION_THRESHOLD):
    """通用人脸识别函数，返回识别结果并保留标注"""
    if frame is None or frame.size == 0:
        return frame, []

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb, 1)
    results = []

    for face in faces:
        shape = shape_predictor(rgb, face)
        descriptor = face_encoder.compute_face_descriptor(rgb, shape)
        current_feature = np.array(descriptor)

        identity = "Unknown"
        min_distance = float("inf")

        # 遍历所有已知人脸特征，找到距离最小且小于阈值的匹配
        for name, saved_feature in name_features.items():
            distance = np.linalg.norm(current_feature - saved_feature)
            if distance < min_distance and distance < threshold:  # 使用传入的threshold参数
                min_distance = distance
                identity = name

        confidence = (1 - min_distance / threshold) * 100 if min_distance < threshold else 0

        results.append({
            "name": identity,
            "confidence": confidence,
            "location": (face.left(), face.top(), face.right(), face.bottom())
        })

        # 保留人脸识别标注（姓名和置信度）
        color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)
        try:
            cv2.putText(frame, f"{identity} ({confidence:.1f}%)",
                        (face.left() + 5, face.top() - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except:
            cv2.putText(frame, f"{identity} ({confidence:.1f}%)",
                        (face.left() + 5, face.top() - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, results


def process_sign_in(frame, threshold=RECOGNITION_THRESHOLD):
    """签到函数，移除签到状态标注"""
    global is_signing_active, signed_names, attendance_records

    if not is_signing_active or frame is None:
        return frame, get_signed_names_text()

    _, faces = recognize_face(frame.copy(), threshold)

    # 处理签到逻辑（保留人脸识别，但移除签到状态文字标注）
    for face in faces:
        if face["name"] != "Unknown" and face["name"] not in signed_names:
            signed_names.add(face["name"])
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            attendance_records.append([face["name"], now])
            print(f"[已签到] {face['name']} —— {now}")

    # 移除签到状态文字标注（仅保留人脸识别标注）
    # 原代码中的 cv2.putText 已删除

    return frame, get_signed_names_text()


def recognize_face_image(image_path, threshold=RECOGNITION_THRESHOLD):
    """图片人脸识别"""
    if not image_path or not os.path.exists(image_path):
        return None, "错误：文件路径不存在"

    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in valid_extensions:
        return None, "错误：仅支持JPG/PNG/BMP格式"

    try:
        frame = cv2.imread(image_path)
        if frame is None:
            return None, "错误：无效图片文件"

        result_frame, faces = recognize_face(frame, threshold)
        if result_frame is not None:
            result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        if not faces:
            return result_frame, "未检测到人脸，请确保人脸清晰且占画面比例足够"

        report = "\n".join([
            f"{i + 1}. {face['name']} ({face['confidence']:.1f}%) - 位置: {face['location']}"
            for i, face in enumerate(faces)
        ])
        return result_frame, report

    except Exception as e:
        return None, f"处理图片时出错：{str(e)}"


def create_app():
    evaluation_images = check_evaluation_images()

    with gr.Blocks(title="人脸识别系统") as app:
        gr.Markdown("# 人脸识别系统")

        with gr.Tabs():
            # 1. 签到系统选项卡
            with gr.Tab("签到系统"):
                with gr.Row():
                    with gr.Column(scale=1):
                        threshold_slider = gr.Slider(0.1, 0.8, RECOGNITION_THRESHOLD, step=0.05, label="识别阈值")
                        status_text = gr.Textbox("就绪", label="系统状态", lines=2)
                        start_sign_btn = gr.Button("开始签到", variant="primary")
                        stop_sign_btn = gr.Button("停止签到", variant="secondary", interactive=False)
                        export_btn = gr.Button("导出签到结果", variant="info", interactive=False)
                        reset_btn = gr.Button("重置签到", variant="secondary")
                        export_status = gr.Textbox("", label="导出状态", lines=1)
                        gr.Markdown("### 已签到人员")
                        signed_names_display = gr.Textbox("", lines=10, interactive=False)
                    with gr.Column(scale=2):
                        webcam = gr.Image(source="webcam", streaming=True, type="numpy", label="摄像头画面")
                        output = gr.Image(label="签到结果", type="numpy")

                def start_signing_fn():
                    global is_signing_active
                    is_signing_active = True
                    return (
                        gr.Button("开始签到", variant="primary", interactive=False),
                        gr.Button("停止签到", variant="secondary", interactive=True),
                        gr.Button("导出签到结果", variant="info", interactive=True),
                        "签到已启动"
                    )

                def stop_signing_fn():
                    global is_signing_active
                    is_signing_active = False
                    return (
                        gr.Button("开始签到", variant="primary", interactive=True),
                        gr.Button("停止签到", variant="secondary", interactive=False),
                        "签到已停止",
                        get_signed_names_text()
                    )

                start_sign_btn.click(start_signing_fn, outputs=[start_sign_btn, stop_sign_btn, export_btn, status_text])
                stop_sign_btn.click(stop_signing_fn,
                                    outputs=[start_sign_btn, stop_sign_btn, status_text, signed_names_display])
                export_btn.click(save_attendance_to_excel, outputs=export_status)
                reset_btn.click(reset_attendance, outputs=[status_text, signed_names_display])
                webcam.stream(process_sign_in, inputs=[webcam, threshold_slider],
                              outputs=[output, signed_names_display])

            # 2. 详细识别结果选项卡
            with gr.Tab("详细识别结果"):
                with gr.Row():
                    with gr.Column(scale=1):
                        threshold_slider_detail = gr.Slider(0.1, 0.8, RECOGNITION_THRESHOLD, step=0.05,
                                                            label="识别阈值")
                        show_fps = gr.Checkbox(False, label="显示帧率")
                        status_text_detail = gr.Textbox("就绪", label="系统状态", lines=2)
                        detected_faces = gr.Textbox("未检测到人脸", label="识别结果", lines=5)
                        face_boxes = gr.Textbox("", label="人脸框位置", lines=5)
                        start_btn = gr.Button("启动识别", variant="primary")
                        stop_btn = gr.Button("停止识别", variant="secondary", interactive=False)
                    with gr.Column(scale=2):
                        webcam_detail = gr.Image(source="webcam", streaming=True, type="numpy", label="摄像头画面")
                        output_detail = gr.Image(label="识别结果", type="numpy")

                is_recognizing = gr.State(False)  # 独立的识别状态
                last_time = gr.State(time.time())

                def start_recognition_fn():
                    is_recognizing.value = True
                    last_time.value = time.time()
                    return (
                        gr.Button("启动识别", variant="primary", interactive=False),
                        gr.Button("停止识别", variant="secondary", interactive=True),
                        "正在识别..."
                    )

                def stop_recognition_fn():
                    is_recognizing.value = False
                    return (
                        gr.Button("启动识别", variant="primary", interactive=True),
                        gr.Button("停止识别", variant="secondary", interactive=False),
                        "已停止"
                    )

                def process_frame_detail_fn(frame, threshold_val, show_fps_val):
                    if not is_recognizing.value or frame is None:
                        return frame, "未检测到人脸", ""
                    result_image, results = recognize_face(frame, threshold_val)
                    text_result = "\n".join([f"{i + 1}. {r['name']} ({r['confidence']:.1f}%)" for i, r in
                                             enumerate(results)]) if results else "未检测到人脸"
                    box_result = "\n".join(
                        [f"{i + 1}. {r['location']}" for i, r in enumerate(results)]) if results else ""
                    if show_fps_val:
                        current_time = time.time()
                        fps = 1 / (current_time - last_time.value) if (current_time - last_time.value) > 0 else 0
                        last_time.value = current_time
                        cv2.putText(result_image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 0), 2)
                    return result_image, text_result, box_result

                start_btn.click(start_recognition_fn, outputs=[start_btn, stop_btn, status_text_detail])
                stop_btn.click(stop_recognition_fn, outputs=[start_btn, stop_btn, status_text_detail])
                webcam_detail.stream(process_frame_detail_fn, inputs=[webcam_detail, threshold_slider_detail, show_fps],
                                     outputs=[output_detail, detected_faces, face_boxes])

            # 3. 图片人脸识别选项卡
            with gr.Tab("图片人脸识别"):
                with gr.Column():
                    gr.Markdown("# 图片人脸识别")
                    with gr.Row():
                        with gr.Column(scale=1):
                            image_input = gr.File(label="上传图片", file_types=["image"])
                            threshold_img_slider = gr.Slider(0.1, 0.8, RECOGNITION_THRESHOLD, step=0.05,
                                                             label="识别阈值")
                            recognize_btn = gr.Button("开始识别", variant="primary")
                            result_text = gr.Textbox("", label="识别报告", lines=10)
                        with gr.Column(scale=2):
                            result_image = gr.Image(label="识别结果", type="numpy")

                    def process_image_fn(file, threshold):
                        if not file:
                            return None, "请上传图片"
                        return recognize_face_image(file.name, threshold)

                    recognize_btn.click(process_image_fn, inputs=[image_input, threshold_img_slider],
                                        outputs=[result_image, result_text])

            # 4. 活体检测选项卡
            with gr.Tab("活体检测"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 检测指引")
                        gr.Markdown("- 眨眼（连续闭眼3次）")
                        gr.Markdown("- 张嘴（保持开口5秒）")
                        gr.Markdown("- 点头/摇头（任意动作5次）")
                        live_status = gr.Textbox("", label="检测状态", lines=3, interactive=False)
                        action_results = gr.Textbox("", label="动作详情", lines=4, interactive=False)
                        start_live_btn = gr.Button("开始检测", variant="primary")
                        stop_live_btn = gr.Button("停止检测", variant="secondary", interactive=False)
                        reset_live_btn = gr.Button("重置检测", variant="secondary")
                    with gr.Column(scale=2):
                        live_webcam = gr.Image(source="webcam", streaming=True, type="numpy", label="实时画面")
                        live_output = gr.Image(label="检测结果", type="numpy")

                # 状态变量
                is_live_active = gr.State(False)
                detection_passed = gr.State(False)
                blink_cnt = gr.State(0)
                mouth_cnt = gr.State(0)
                head_cnt = gr.State(0)
                success_count = gr.State(0)
                compare_point = gr.State([0, 0])
                size = gr.State((480, 640))

                def live_detection(frame):
                    if not is_live_active.value or frame is None:
                        if detection_passed.value:
                            return frame, "✅ 活体检测通过！", "请点击'重置检测'重新开始"
                        return frame, "", ""

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = detector(frame_rgb, 0)
                    size.value = frame.shape[:2]

                    if len(faces) != 1:
                        compare_point.value = [0, 0]
                        return frame, "请确保仅1人正对摄像头", ""

                    face = faces[0]
                    landmarks = np.array([[p.x, p.y] for p in shape_predictor(frame_rgb, face).parts()])
                    action_details = []

                    # 眨眼检测
                    left_eye = landmarks[36:42]
                    right_eye = landmarks[42:48]
                    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
                    if ear < 0.2:
                        blink_cnt.value += 1
                        if blink_cnt.value >= 3:
                            blink_status = "✅ 眨眼成功"
                            if success_count.value < 3:
                                success_count.value += 1
                            blink_cnt.value = 0
                        else:
                            blink_status = f"⏳ 眨眼进度: {blink_cnt.value}/3"
                    else:
                        blink_status = "🔵 等待眨眼"
                    action_details.append(blink_status)

                    # 张嘴检测
                    mouth = landmarks[48:68]
                    mar = mouth_aspect_ratio(mouth)
                    if mar > 0.5:
                        mouth_cnt.value += 1
                        if mouth_cnt.value >= 5:
                            mouth_status = "✅ 张嘴成功"
                            if success_count.value < 3:
                                success_count.value += 1
                            mouth_cnt.value = 0
                        else:
                            mouth_status = f"⏳ 张嘴进度: {mouth_cnt.value}/5"
                    else:
                        mouth_status = "🔵 等待张嘴"
                    action_details.append(mouth_status)

                    # 头部动作检测
                    nose_points = landmarks[27:36]
                    nose_center = nose_points.mean(axis=0).tolist()
                    prev_point = compare_point.value if compare_point.value else [0, 0]

                    if prev_point != [0, 0]:
                        nod_val = nod_aspect_ratio(size.value, prev_point, nose_center)
                        shake_val = shake_aspect_ratio(size.value, prev_point, nose_center)
                        if nod_val > 0.03 or shake_val > 0.03:
                            head_cnt.value += 1
                            if head_cnt.value >= 5:
                                head_status = "✅ 头部动作成功"
                                if success_count.value < 3:
                                    success_count.value += 1
                                head_cnt.value = 0
                            else:
                                head_status = f"⏳ 头部动作进度: {head_cnt.value}/5"
                        else:
                            head_status = "🔵 等待头部动作"
                    else:
                        head_status = "🔵 等待头部动作"
                    action_details.append(head_status)

                    # 综合结果
                    live_status_msg = "进行中... 需要完成2种动作"
                    if success_count.value >= 2:
                        live_status_msg = "✅ 活体检测通过！"
                        detection_passed.value = True
                        is_live_active.value = False
                        blink_cnt.value = mouth_cnt.value = head_cnt.value = 0
                        success_count.value = 0

                    action_details_text = "\n".join(action_details)

                    # 仅保留人脸框标注，移除所有文字标注
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                    # 移除 cv2.putText 调用，不再显示状态文字

                    compare_point.value = nose_center
                    return frame, live_status_msg, action_details_text

                def start_live():
                    is_live_active.value = True
                    detection_passed.value = False
                    blink_cnt.value = 0
                    mouth_cnt.value = 0
                    head_cnt.value = 0
                    success_count.value = 0
                    compare_point.value = [0, 0]
                    return (
                        gr.Button("开始检测", variant="primary", interactive=False),
                        gr.Button("停止检测", variant="secondary", interactive=True),
                        "检测已启动",
                        ""
                    )

                def stop_live():
                    is_live_active.value = False
                    return (
                        gr.Button("开始检测", variant="primary", interactive=True),
                        gr.Button("停止检测", variant="secondary", interactive=False),
                        "检测已停止",
                        ""
                    )

                def reset_live():
                    is_live_active.value = False
                    detection_passed.value = False
                    blink_cnt.value = 0
                    mouth_cnt.value = 0
                    head_cnt.value = 0
                    success_count.value = 0
                    compare_point.value = [0, 0]
                    return "检测状态已重置", ""

                start_live_btn.click(
                    start_live,
                    outputs=[start_live_btn, stop_live_btn, live_status, action_results]
                )
                stop_live_btn.click(
                    stop_live,
                    outputs=[start_live_btn, stop_live_btn, live_status, action_results]
                )
                reset_live_btn.click(reset_live, outputs=[live_status, action_results])
                live_webcam.stream(
                    live_detection,
                    inputs=[live_webcam],
                    outputs=[live_output, live_status, action_results]
                )

            # 5. 系统参数设置介绍
            with gr.Tab("系统参数设置"):
                gr.Markdown("# 系统参数设置说明")

                # 显示两张评估图片
                gr.Markdown("## 系统评估指标")
                with gr.Row():
                    with gr.Column():
                        metrics_image = gr.Image(
                            value=evaluation_images["metrics"],
                            label="系统评估指标",
                            type="filepath",
                            interactive=False
                        )
                    with gr.Column():
                        confusion_matrix = gr.Image(
                            value=evaluation_images["confusion_matrix"],
                            label="混淆矩阵",
                            type="filepath",
                            interactive=False
                        )

                # 超参数设置
                gr.Markdown("## 超参数设置")
                gr.Markdown("""
                系统提供了以下可调整的超参数，用户可根据实际需求进行调整：

                ### 人脸识别阈值 (`RECOGNITION_THRESHOLD`)
                - **默认值**：0.45
                - **范围**：0.1 - 0.8
                - **影响**：
                  - 阈值越小，识别要求越严格，误识率降低，但可能增加拒识率
                  - 阈值越大，识别要求越宽松，识别率提高，但可能增加误识率
                - **调整建议**：
                  - 在安全要求较高的场景下，可适当降低阈值（如0.4）
                  - 在需要高识别率的场景下，可适当提高阈值（如0.5）

                ### 活体检测参数
                - **眨眼检测**：
                  - 眼睛纵横比阈值：0.2
                  - 触发次数：连续眨眼3次
                - **张嘴检测**：
                  - 嘴巴纵横比阈值：0.5
                  - 触发持续时间：保持张嘴5秒
                - **头部动作检测**：
                  - 点头/摇头阈值：0.03
                  - 触发次数：任意动作5次

                ### 系统性能参数
                - **摄像头帧率控制**：
                  - 默认启用自适应帧率
                  - 可通过界面复选框启用FPS显示
                """)

                # 模型参数设置
                gr.Markdown("## 模型参数设置")
                gr.Markdown("""
                系统使用的主要模型及其参数配置：

                ### 人脸检测器 (`dlib.get_frontal_face_detector()`)
                - **模型类型**：基于HOG特征和SVM的人脸检测器
                - **特点**：
                  - 对正面和近正面人脸检测效果较好
                  - 计算效率高，适合实时应用
                - **参数调整**：
                  - 检测器缩放因子：默认值为1（可通过`detector(rgb, 1)`中的第二个参数调整）

                ### 人脸特征提取器 (`dlib.face_recognition_model_v1`)
                - **模型类型**：基于ResNet的128维人脸特征提取器
                - **特点**：
                  - 提取的特征具有较高的区分性
                  - 对光照、表情变化有较好的鲁棒性
                - **依赖文件**：
                  - `shape_predictor_68_face_landmarks.dat`：68点人脸特征点检测器
                  - `dlib_face_recognition_resnet_model_v1.dat`：人脸特征提取模型

                ### 特征匹配方法
                - **距离度量**：欧氏距离
                - **匹配逻辑**：
                  - 计算输入人脸特征与数据库中特征的欧氏距离
                  - 距离小于设定阈值的认为匹配成功
                """)

                # 仅保留参数调整步骤
                gr.Markdown("## 参数调整指南")
                gr.Markdown("""
                ### 调整步骤
                1. 在相应功能界面中找到参数调整滑块或选项
                2. 逐步调整参数并观察系统表现
                3. 保存最佳参数配置
                """)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)