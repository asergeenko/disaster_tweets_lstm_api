from fastapi import FastAPI
import uvicorn

from core import predict, get_model

model = get_model()
app = FastAPI()

@app.get('/')
def get_root():
    return {'message': 'Welcome to the disaster tweets prediction API'}


@app.get("/predict/{message}")
async def get_predict(message):
    return {'message':message, 'disaster_tweet':predict(message, model)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



