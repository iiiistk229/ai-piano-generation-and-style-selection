from transformers import pipeline
import scipy
import joblib
import librosa
import numpy as np

# 加载生成模型
generator = pipeline("text-to-audio", model="facebook/musicgen-small")

# 加载你训练好的风格模型
model = joblib.load("style_model.pkl")

# 提取特征函数（和训练时一致）
def extract_features(audio, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)
    return mfcc.reshape(1, -1)

# 输入控制
emotion = input("情绪 (happy/sad): ")
instrument = input("乐器 (piano/guitar): ")
tempo = input("速度 (fast/slow): ")

prompt = f"{emotion} {instrument} music, {tempo} tempo"

best_score = -1
best_audio = None

print("开始生成多段音乐...")

# 生成5次
for i in range(5):
    print(f"生成第{i+1}段...")
    
    music = generator(prompt, forward_params={"do_sample": True})
    
    audio = music["audio"]
    sr = music["sampling_rate"]
    
    # 提取特征
    features = extract_features(audio, sr)
    
    # 预测风格（1=目标风格）
    score = model.predict_proba(features)[0][1]
    
    print(f"风格匹配分数: {score}")
    
    if score > best_score:
        best_score = score
        best_audio = audio

# 保存最好的
scipy.io.wavfile.write("best_music.wav", rate=sr, data=best_audio)

print("完成！已保存 best_music.wav")