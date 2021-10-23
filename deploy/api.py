from fastapi import FastAPI, File, UploadFile
import uvicorn


from core import predict, get_model,train_pipeline, logging

model = get_model()
app = FastAPI()

@app.get('/')
def get_root():
    return {'message': 'Welcome to the disaster tweets prediction API'}

@app.post("/train/")
def post_train(csv_file: UploadFile = File(...)):
    global model
    try:
        model = train_pipeline(csv_file.file)
        message = 'Model has trained succesfully'
    except Exception as e:
        message = 'Error during model training. %s'%(str(e))
        logging.exception(message)
    return {'message':message}

@app.get("/predict/{message}")
async def get_predict(message):
    return {'message':message, 'disaster_tweet':predict(message, model)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



