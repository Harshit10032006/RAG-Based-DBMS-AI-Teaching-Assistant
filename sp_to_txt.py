import whisper 
import json


model = whisper.load_model("large")

result=model.transcribe(audio='Audio(SQL)/output.mp3',language='hi',task='translate') # audio language 

# res=result

# with open ('test.','w') as f :
#     json.dump(f,res)


print(result)



    