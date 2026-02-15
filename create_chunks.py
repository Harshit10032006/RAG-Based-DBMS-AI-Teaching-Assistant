import os 
import whisper
import os 
import json

model=whisper.load_model('large')

audios=os.listdir('Audio(SQL)')


for audio in audios :
    number=audio.split('：')[0].split('Lecture')[1]
    Title=audio.split('：')[1].split('.mp3')[0]
    
    result=model.transcribe(audio=f'Audio(SQL)/{audio}',language='hi',task='translate')
    




    chunks=[]

    for segment in result['segments']:
        chunks.append({'number':number,'title':Title,'start':segment['start'],'end':segment['end'],'text':segment['text']})

    chunks_withdata={'chunks':chunks,'text':result['text']}
    
    with open (f'Chunks/{audio}.json','w') as f :
     json.dump(chunks_withdata,f)