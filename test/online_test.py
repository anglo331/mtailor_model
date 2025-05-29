import requests
import json

API_URL = "https://api.cortex.cerebrium.ai/v4/p-f70cf007/mtailor/predict"




import requests
import base64
import os
import json

# Configuration for local testing

def send_prediction_request(image_path: str):
    print(f"\n--- Testing with image: {image_path} ---")
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}. Skipping test.")
        return

    try:
        # Read the image file in binary mode and encode it to base64
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # Prepare the request headers and payload
        headers = {
        'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLWY3MGNmMDA3IiwibmFtZSI6IiIsImRlc2NyaXB0aW9uIjoiIiwiZXhwIjoyMDY0MDcyMDYxfQ.udGQqHy1ooWRKEZVxJA7-0kflu4-3Ho7dsDKF9Jh5db6NBG2QOaZstXqc6h37ykaQj8FxZX3941sybyMesfVpdk_B7v-edkcawefgXXqq0ywRpVls3NfJyKcWq-Mh82SLwOmnlxRwOm-A2oozmYZBwlnAysyHABzWW3A8DI_pV136qHSCnNI9PqifysXtGgnN42ozyoxdmIo2Iu-K9LgnE2Angt_FfLKFdOG_R0bDcEA0VjNMZoFNDBylwEY1TsewoAySqavdvfwJdsVEe-S6WBlAOfa_X2qudR9EdEEcKeoQM0zIfnMgzLQYNZc4yxz11clfL17IuDLVYlk3aHz3Q',
        'Content-Type': 'application/json'
        }
        payload = {"image_base64": encoded_string}

        print(f"Sending POST request to {API_URL}...")
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))

        print(f"Response Status Code: {response.status_code}")
        response_json = response.json()
        print("Response Body:")
        print(json.dumps(response_json, indent=2)) # Pretty print JSON

        if response.status_code == 200:
            predicted_id = response_json.get("predicted_class_id")
            confidence = response_json.get("confidence")
            print(f"\nPrediction successful!")
            print(f"Predicted Class ID: {predicted_id}")
            print(f"Confidence: {confidence:.4f}")

            # Verify expected predictions (as per your assignment)
            if "n01440764_tench.jpeg" in image_path and predicted_id == 0:
                print("✅ Correct prediction for tench!")
            elif "n01667114_mud_turtle.jpeg" in image_path and predicted_id == 35:
                print("✅ Correct prediction for mud turtle!")
            else:
                print("⚠️ Prediction does not match expected ID or image name. Please verify manually.")
        else:
            print(f"❌ Prediction failed with error: {response_json.get('detail') or response_json.get('error')}")

    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Could not connect to the API at {API_URL}.")
        print("Please ensure your FastAPI app is running (uvicorn app:app --host 0.0.0.0 --port 8000).")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

print("\n" + "="*60 + "\n") # Separator for readability

if __name__ == "__main__":
    # Test with the tench image
    send_prediction_request("src/n01440764_tench.jpeg")
    
    # Test with the mud turtle image
    send_prediction_request("src/n01667114_mud_turtle.jpeg")
    
    # Test with a non-existent image to check error handling
    send_prediction_request("mtailor_model/test/images/non_existent_image.jpeg")