import whisper

model = whisper.load_model("small")

# 假设音频文件路径为 'audio.wav'
#result = model.transcribe("audio.wav")
result = model.transcribe("audio2.m4a", language="zh")

print(result["text"])
