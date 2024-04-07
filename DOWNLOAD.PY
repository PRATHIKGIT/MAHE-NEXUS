import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("DATEXIS/CORe-clinical-diagnosis-prediction")
model = AutoModelForSequenceClassification.from_pretrained("DATEXIS/CORe-clinical-diagnosis-prediction")

# Function to perform inference
def perform_inference(input_text):
    # Tokenize the input text
    tokenized_input = tokenizer(input_text, return_tensors="pt")
    
    # Perform inference using the model
    with torch.no_grad():
        output = model(**tokenized_input)
    
    # Apply sigmoid activation to the logits
    predictions = torch.sigmoid(output.logits)
    
    # Define a threshold for classification
    threshold = 0.3
    
    # Get predicted labels based on the threshold
    predicted_labels = [model.config.id2label[_id] for _id in (predictions > threshold).nonzero()[:, 1].tolist()]
    
    return predicted_labels

# Streamlit app
def main():
    st.title("Clinical Diagnosis Prediction")
    
    # Input text area for user input
    input_text = st.text_area("Enter the clinical text:", "")
    
    # File path input for CSV file
    csv_file_path =r"C:\Users\prath\Downloads\CODING\Healthsync\CORE\CMS32_DESC_LONG_SHORT_DX.xlsx"
    csv_file_path1 =r"C:\Users\prath\Downloads\CODING\Healthsync\CORE\CMS32_DESC_LONG_SHORT_SG.xlsx"

    
    if csv_file_path.strip() != "":
        try:
            # Load CSV file
            df = pd.read_excel(csv_file_path)
            df1=pd.read_excel(csv_file_path1)
            
            # Button to trigger prediction
            if st.button("Predict"):
                if input_text.strip() != "":
                    # Perform inference
                    predicted_labels = perform_inference(input_text)

                    st.success("Predicted Labels:")
                    st.write(predicted_labels)
                    
                    # Search for predicted codes in CSV file
                    matched_codes = df[df['DIAGNOSIS CODE'].astype(str).isin(predicted_labels)]
                    matched_codes1 = df1[df1['PROCEDURE CODE'].astype(str).isin(predicted_labels)]
                    
                    
                    if not matched_codes.empty:
                        st.success("Matched Procedure Codes:")
                        st.write(matched_codes[['DIAGNOSIS CODE', 'LONG DESCRIPTION', 'SHORT DESCRIPTION']])
                        st.write(matched_codes1[['PROCEDURE CODE', 'LONG DESCRIPTION', 'SHORT DESCRIPTION']])
                    else:
                        st.warning("No matching procedure codes found.")
                else:
                    st.warning("Please enter some text for prediction.")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
