from fastapi import FastAPI, Request
import uvicorn
import os
from dotenv import load_dotenv
from agent import PoliticBotAgent
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

politicbot = PoliticBotAgent()

class Request(BaseModel):
    request: str

@app.get("/")
def read_root():
    return {"Halo": "2024"}

@app.post("/politic-bot/")
async def answer(request: Request):
    try:
        question = request.request
        # Assuming async_generate expects a string input
        response = politicbot.async_generate(str(question))

        # Check if the response is a string and return as expected
        if isinstance(response, str):
            return {"message": response}
        else:
            return {"message": "Unexpected response type"}

    except Exception as e:
        return {"message": "Error nya ini: "+str(e)}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)