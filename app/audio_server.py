import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

import torch
import torchaudio
import pickle

export_file_url = 'https://www.dropbox.com/s/6bgq8t6yextloqp/export.pkl?raw=1'
export_file_name = 'audio_model.pkl'

classes = ['tank','pistal','machine_gun','ship','aircraft']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        with open(path / export_file_name, 'rb') as file:
            learn = pickle.load( file)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())

def format_sound_data(soundData):
    clip_length = 16000
    tempData = torch.zeros(clip_length)  # tempData accounts for audio clips that are too short
    start = next((idx for idx, obj in enumerate(soundData) if obj > 0.1), 0)
    soundData = soundData[start:]
    if soundData.numel() < clip_length:
        tempData[:soundData.numel()] = soundData[:]
    else:
        tempData[:] = soundData[:clip_length]

    soundData = tempData
    soundFormatted = torch.zeros([int(clip_length / 5)])
    soundFormatted[:int(clip_length / 5)] = soundData[::5]  # take every fifth sample of soundData
    # soundFormatted = soundFormatted.permute(1, 0)
    return soundFormatted


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    audio_data = await request.form()
    audio_bytes = await (audio_data['file'].read())
    with open("temp.mp3",'wb') as out:
        out.write(audio_bytes)
    print(type(audio_bytes))
    print(type(BytesIO(audio_bytes)))
    raw_audio = torch.mean(torchaudio.load("temp.mp3", out = None, normalization = True)[0],0)
    audio = format_sound_data(raw_audio)
    prediction = learn.predict([audio.t().numpy()])[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5001, log_level="info")
