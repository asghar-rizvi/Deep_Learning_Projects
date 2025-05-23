from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from werkzeug.utils import secure_filename
from models_Handling import ModelHandler
from wordModel import HandwritingRecognizer

app = FastAPI()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.character_handler = ModelHandler()  
    app.state.word_model_handler = HandwritingRecognizer()
    yield

app = FastAPI(lifespan=lifespan)

def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_type: str = Form('cnn')):
    if not file.filename:
        return JSONResponse({'error': 'No selected file'}, status_code=400)
    
    if not allowed_file(file.filename):
        return JSONResponse({'error': 'Invalid file type'}, status_code=400)
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())
        
        model_type = model_type.lower()
        if model_type == 'ml':
            result = character_handler.ml_predict(filepath)
        elif model_type == 'word':
            result = word_model_handler.predict(filepath)
        else:  
            result = character_handler.cnn_predict(filepath)

        os.remove(filepath)
        
        return JSONResponse({
            'prediction': result,
            'model_used': model_type
        })
        
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)