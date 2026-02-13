import whisper 
import json


model = whisper.load_model("large")

result=model.transcribe(audio='Audio(SQL)/output.mp3',language='hi',task='translate') # audio language 






chunks=[]

for segment in result['segments']:
    chunks.append({'start':segment['start'],'end':segment['end'],'text':segment['text']})

print(chunks)



with open ('test.json','w') as f :
    json.dump(chunks,f)


    