from fastapi import FastAPI
from pydantic import BaseModel
from body_classifier import BodyNonbodyClassifier

#TODO: Add validations and error cases

class InputText(BaseModel):
    text: str

cleaner = BodyNonbodyClassifier()
app = FastAPI()

@app.post("/clean")
def classify_noise(input_text: InputText):
    output = cleaner.predict(input_text.text)
    return output