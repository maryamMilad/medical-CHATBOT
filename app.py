import onnxruntime as ort
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import logging

app = FastAPI()

# Load the ONNX model
ort_session = ort.InferenceSession("medical_chatbot_model.onnx")

# Load the tokenizer (assuming a Hugging Face model)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Adjust this to your specific tokenizer

# Define input data structure for request
class InputData(BaseModel):
    text: str

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict(data: InputData):
    try:
        # Log the received text
        logger.info(f"Received request with text: {data.text}")
        
        # Tokenize and encode the input text
        inputs = tokenizer.encode_plus(
            data.text,
            add_special_tokens=True,
            return_tensors="np",  # Return numpy arrays
            padding="max_length",
            max_length=512,  # Adjust based on your model's requirements
            truncation=True
        )

        # Convert tokenized input to numpy array for ONNX
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Prepare inputs for the ONNX model
        ort_inputs = {
            ort_session.get_inputs()[0].name: input_ids,
            ort_session.get_inputs()[1].name: attention_mask,
        }

        # Run inference on the ONNX model
        ort_outputs = ort_session.run(None, ort_inputs)

        # Process the outputs (assuming binary classification)
        predictions = np.argmax(ort_outputs[0], axis=1).tolist()
        labels = ["negative", "positive"]  # Example labels for binary classification
        label = labels[predictions[0]]

        # Return the prediction result
        return {"prediction": label}

    except Exception as e:
        # Handle any errors that occur during the process
        logger.error(f"Error occurred: {str(e)}")
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI NLP model!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import logging

logging.basicConfig(level=logging.INFO)

@app.on_event("startup")
async def startup_event():
    logging.info("Application startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Application shutdown complete.")