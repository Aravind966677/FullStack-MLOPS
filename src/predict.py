from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import yaml
from utils import format_input_data, prepare_data
from typing import Dict

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(title="Churn Prediction API")

# Load model and scaler
try:
    model = joblib.load(config['model_path'])
    scaler = joblib.load(config['scaler_path'])
except Exception as e:
    raise RuntimeError(f"Error loading model or scaler: {str(e)}")

class PredictionInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

class PredictionOutput(BaseModel):
    churn_probability: float
    prediction: bool
    prediction_explanation: str

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Make a churn prediction for a single customer.
    """
    try:
        # Convert input data to DataFrame
        df = format_input_data(input_data.dict())
        
        # Prepare features
        X, _ = prepare_data(df, scaler)
        
        # Make prediction
        probability = model.predict_proba(X)[0][1]
        prediction = probability >= 0.5
        
        # Create explanation
        explanation = generate_explanation(input_data, probability)
        
        return {
            "churn_probability": float(probability),
            "prediction": bool(prediction),
            "prediction_explanation": explanation
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def generate_explanation(input_data: PredictionInput, probability: float) -> str:
    """
    Generate a human-readable explanation of the prediction.
    """
    risk_level = "high" if probability >= 0.7 else "moderate" if probability >= 0.3 else "low"
    
    explanation = f"This customer has a {risk_level} risk of churning ({probability:.1%} probability). "
    
    # Add contract-specific information
    if input_data.Contract == "Month-to-month":
        explanation += "The month-to-month contract type increases churn risk. "
    
    # Add tenure-specific information
    if input_data.tenure < 12:
        explanation += "The short tenure period suggests higher churn risk. "
    elif input_data.tenure > 24:
        explanation += "The long tenure period suggests lower churn risk. "
    
    # Add service-related information
    if input_data.InternetService == "Fiber optic":
        if input_data.OnlineSecurity == "No":
            explanation += "Fiber optic service without online security shows higher churn rates. "
    
    # Add streaming-related information
    if input_data.StreamingTV == "Yes" and input_data.StreamingMovies == "Yes":
        explanation += "Customers with multiple streaming services tend to be more engaged. "
    
    return explanation

@app.get("/health")
async def health_check():
    """
    Check if the service is healthy.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config['api']['host'], port=config['api']['port'])