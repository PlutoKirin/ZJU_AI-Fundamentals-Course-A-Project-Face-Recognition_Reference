import os
import cv2
import dlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, recall_score, f1_score, precision_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from itertools import cycle
import seaborn as sns
from collections import defaultdict

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")


def save_metrics_image(metrics_dict, save_path):
    """将评估指标保存为图片"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')

    # 创建表格数据
    cell_text = [[f"{v:.4f}"] for v in metrics_dict.values()]
    columns = ["Value"]

    # 绘制表格
    table = ax.table(
        cellText=cell_text,
        rowLabels=list(metrics_dict.keys()),
        colLabels=columns,
        cellLoc='center',
        loc='center'
    )
    table.scale(1, 2)
    table.set_fontsize(14)

    plt.title("Model Evaluation Metrics", pad=20)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
def extract_face_features(dataset_dir, test_ratio):
    """
    从数据集中提取人脸特征并划分训练集和测试集
    """
    name_features = {}
    test_features = defaultdict(list)
    train_features = defaultdict(list)

    # 遍历每个人物文件夹
    for name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, name)
        if not os.path.isdir(person_dir):
            continue

        features = []
        image_count = 0
        # 处理每人照片
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 转为RGB格式并检测人脸
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces1 = detector(rgb, 1)
            if len(faces1) == 0:
                continue  # 跳过无人脸的图片
            else :
                faces = detector(rgb,1)
            # 提取人脸特征
            shape = shape_predictor(rgb, faces[0])
            if shape.num_parts < 68 :
                continue
            face_descriptor = face_encoder.compute_face_descriptor(rgb, shape)
            features.append(np.array(face_descriptor))
            image_count += 1

        # 划分训练集和测试集
        if features:
            if len(features)>1 :
                split_idx = int(len(features) * (1 - test_ratio))
                train_features[name] = features[:split_idx]
                test_features[name] = features[split_idx:]
                print(

                    f"[Success] Features for {name} extracted (Training set: {len(train_features[name])} images, Test set: {len(test_features[name])} images)")
            else :
                train_features[name] = features[0:1]
                test_features[name] = features[0:1]
                print(
                    f"[Success] Features for person {name} have been extracted (Training set: {len(train_features[name])} images, Test set: {len(test_features[name])} images)"
                )
    # 计算训练集的平均特征向量
    for name, features in train_features.items():
        if len(features) == 0 :
            continue
        name_features[name] = np.mean(features, axis=0)

    # 保存特征数据
    with open("face_features.pkl", "wb") as f:
        pickle.dump(name_features, f)
    print("\nTraining completed! The feature data has been saved as face_features.pkl")

    return name_features, test_features


def evaluate_model(name_features, test_features, threshold=0.45,output_dir="evaluation"):
    """
    评估人脸识别模型性能并保存结果图片

    参数:
    name_features -- 训练集平均特征向量 {name: feature_vector}
    test_features -- 测试集特征 {name: [feature_vector1, ...]}
    output_dir -- 结果保存目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 准备测试数据
    X_test = []
    y_true = []
    class_names = sorted(name_features.keys())

    # 构建测试数据集
    for name, features in test_features.items():
        if name not in name_features:  # 跳过训练集中没有的类别
            continue
        X_test.extend(features)
        y_true.extend([name] * len(features))

    if not X_test:
        print("Error: No valid test data")
        return

    # 转换为numpy数组
    X_test = np.array(X_test)
    y_true = np.array(y_true)

    # 预测标签和计算置信度
    y_pred = []
    y_score = np.zeros((len(X_test), len(class_names)))

    # 计算每个样本到每个类中心的距离
    for i, sample in enumerate(X_test):
        distances = []
        for cls in class_names:
            dist = np.linalg.norm(sample - name_features[cls])
            distances.append(dist)

        # 转换为置信度分数 (距离越小置信度越高)
        distances = np.array(distances)
        scores = 1 / (1 + distances)  # 将距离转换为[0,1]区间的分数
        y_score[i] = scores

        # 预测标签 (取最小距离)
        pred_idx = np.argmin(distances)
        y_pred.append(class_names[pred_idx])

    y_pred = np.array(y_pred)

    # 1. 计算并保存分类报告
    report = classification_report(y_true, y_pred, target_names=class_names,zero_division=0)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # 2. 绘制并保存混淆矩阵
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # 3. 绘制并保存ROC曲线 (多类别)
    plt.figure(figsize=(10, 8))

    # 二值化标签
    y_true_bin = label_binarize(y_true, classes=class_names)
    n_classes = y_true_bin.shape[1]

    # 计算每个类的ROC曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算微平均ROC曲线
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # 绘制所有ROC曲线
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red',
                    'purple', 'pink', 'brown', 'gray', 'olive'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label='ROC {0} (AUC = {1:0.2f})'
                       ''.format(class_names[i], roc_auc[i]))

    # 绘制微平均ROC曲线
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average (AUC = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right", prop={'size': 8})
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

    # 4. 计算并保存准确率
    accuracy = np.mean(y_true == y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    with open(os.path.join(output_dir, "accuracy.txt"), "w") as f:
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")

    print(f"Evaluation completed! Results have been saved to the {output_dir} directory")
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Threshold": threshold
    }
    save_metrics_image(metrics, os.path.join(output_dir, "metrics.png"))


# 使用示例
if __name__ == "__main__":
# 假设已经通过extract_face_features获取了特征
    name_features, test_features = extract_face_features("Face_Detection_Dataset", 0.2)

# 评估模型
    results = evaluate_model(name_features, test_features)