from fastapi import FastAPI, Request
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
# from pydantic import BaseModel
import uvicorn
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import uuid

import socket
from contextlib import closing

import socket

def find_available_port():
    # Create a socket object
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind to a random available port
    sock.bind(('localhost', 0))

    # Get the assigned port
    port = sock.getsockname()[1]

    # Close the socket
    sock.close()

    return port

# Call the function to find an available port
available_port = find_available_port()
print("Available port:", available_port)


app = FastAPI()
model_location = ""
tokenizer_location = ""


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/hello")
def read_root():
    return {"Hello": "Hello"}


def get_model():
    tokenizer = AutoTokenizer.from_pretrained("CarnivoraCanis/berturk-cased-tr-fakenews")
    model = AutoModelForSequenceClassification.from_pretrained("CarnivoraCanis/berturk-cased-tr-fakenews")

    return tokenizer, model

d = {

    1: 'True',
    0: 'Fake'
}

tokenizer, model = get_model()


@app.post("/predict")
async def read_root(request: Request):
    data = await request.json()
    print(data)
    if 'text' in data:
        text = data["text"]
        encoding = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        encoding = {k: v.to(model.device) for k, v in encoding.items()}

        outputs = model(**encoding)

        logits = outputs.logits

        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        probs = probs.detach().numpy()
        label = np.argmax(probs, axis=-1)
        response = {"id": uuid.uuid4(),
                    "data": str(data["text"]),
                    "label": int(label),
                    "prob": float(probs[label])
                    }


    else:
        response = {"Recieved Text": "No Text Found"}
    return response


if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=available_port, reload=True)