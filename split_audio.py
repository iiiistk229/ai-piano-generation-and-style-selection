from pydub import AudioSegment
import os

# 原始音乐路径（放3-4分钟音乐）
INPUT_ROOT = "raw_music"

# 输出路径（切好的30秒数据）
OUTPUT_ROOT = "dataset"

# 每段长度（30秒）
SEGMENT_MS = 30 * 1000


for label in ["target_style", "other_style"]:
    input_folder = os.path.join(INPUT_ROOT, label)
    output_folder = os.path.join(OUTPUT_ROOT, label)

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    count = 0

    # 遍历原始音乐
    for file in os.listdir(input_folder):
        if file.lower().endswith((".mp3", ".wav", ".m4a", ".flac")):

            file_path = os.path.join(input_folder, file)
            print(f"正在处理：{file}")

            audio = AudioSegment.from_file(file_path)

            # 按30秒切片
            for start in range(0, len(audio), SEGMENT_MS):
                segment = audio[start:start + SEGMENT_MS]

                # 只保留长度>=20秒的片段
                if len(segment) >= 20 * 1000:
                    output_name = f"{label}_{count}.wav"
                    output_path = os.path.join(output_folder, output_name)

                    segment.export(output_path, format="wav")
                    count += 1

    print(f"{label} 切片完成，共生成 {count} 段\n")

print("全部音频切片完成")