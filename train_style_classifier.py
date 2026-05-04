import os
import librosa
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

DATASET_PATH = "dataset"

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        return mfcc
    except Exception as e:
        print("读取失败：", file_path, e)
        return None

X = []
y = []

for label_name, label_value in [("target_style", 1), ("other_style", 0)]:
    folder = os.path.join(DATASET_PATH, label_name)

    print("正在读取文件夹：", folder)

    for file in os.listdir(folder):
        if file.lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
            file_path = os.path.join(folder, file)
            print("读取：", file_path)

            features = extract_features(file_path)

            if features is not None:
                X.append(features)
                y.append(label_value)

print("数据量：", len(X))

if len(X) < 2:
    raise ValueError("数据太少或没有成功读取音频，请检查 dataset 文件夹。")

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("准确率：", accuracy)

joblib.dump(model, "style_model.pkl")
print("模型已保存：style_model.pkl")