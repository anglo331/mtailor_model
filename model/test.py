import os
import numpy as np
from PIL import Image
# Import the necessary functions from your model.py
# Assuming model.py is in 'mtailor_model/model/' relative to where this test.py is run
from model import create_session, predict, preprocess_input

# --- Configuration Section ---
# This is where you define important paths and expected values.
# Ensure these paths are correct relative to where you run the test.py script
ONNX_MODEL_PATH = "mtailor_model/model/mtailor_model.onnx"
TEST_IMAGES_DIR = "mtailor_model/src"

# We know these images belong to specific classes in the ImageNet dataset.
# This helps us check if our model predicts correctly.
# Class ID 0: tench, Tinca tinca (a type of fish)
# Class ID 35: mud turtle, loggerhead musk turtle (a type of turtle)
EXPECTED_PREDICTIONS = {
    "n01440764_tench.jpeg": 0,
    "n01667114_mud_turtle.jpeg": 35,
}

# --- Helper Function for Reporting Results ---
# This function simply helps us print "PASSED" or "FAILED" messages.
def report_test_status(test_name, status):
    if status:
        print(f"✅ {test_name}: PASSED")
    else:
        print(f"❌ {test_name}: FAILED")

# --- Test 1: Image Preprocessing ---
# This test checks if your preprocess_input function is correctly preparing images for the model.
def run_preprocessing_test():
    print("\n--- Starting Test: Image Preprocessing ---")
    # Define the path to one of our test images
    test_image_path = os.path.join(TEST_IMAGES_DIR, "n01440764_tench.jpeg")

    # Check if the test image actually exists
    if not os.path.exists(test_image_path):
        print(f"ERROR: Test image not found at {test_image_path}. Please make sure '{TEST_IMAGES_DIR}' folder exists and has the images.")
        report_test_status("Image Preprocessing", False)
        return False

    try:
        # Call the preprocess_input function from model.py
        print(f"  Attempting to preprocess image: {test_image_path}")
        processed_list = preprocess_input(test_image_path)

        # Check 1: Is the output a Python list? (As per preprocess_input returning .tolist())
        if not isinstance(processed_list, list):
            print("  FAIL: Preprocessed output is not a Python list.")
            report_test_status("Image Preprocessing", False)
            return False

        # Convert the list to a NumPy array for easier shape and type checking
        processed_array = np.array(processed_list)

        # Check 2: Does it have the correct shape? (Batch, Channels, Height, Width)
        # Our model expects (1, 3, 224, 224)
        expected_shape = (1, 3, 224, 224)
        if processed_array.shape != expected_shape:
            print(f"  FAIL: Incorrect shape. Expected {expected_shape}, got {processed_array.shape}")
            report_test_status("Image Preprocessing", False)
            return False

        # Check 3: Is the data type correct? Models typically need float32.
        if processed_array.dtype != np.float32:
            print(f"  FAIL: Incorrect data type. Expected float32, got {processed_array.dtype}")
            report_test_status("Image Preprocessing", False)
            return False

        # Check 4: Are the pixel values normalized?
        # After normalization, values should typically be around -2 to 2 (not 0-255 or 0-1).
        if not (np.min(processed_array) >= -5.0 and np.max(processed_array) <= 5.0):
             print(f"  FAIL: Pixel values out of expected range after normalization.")
             print(f"  Min value: {np.min(processed_array):.4f}, Max value: {np.max(processed_array):.4f}")
             report_test_status("Image Preprocessing", False)
             return False

        report_test_status("Image Preprocessing", True)
        return True

    except Exception as e:
        # If any error occurs during preprocessing, catch it and report failure.
        print(f"  An unexpected error occurred during preprocessing: {e}")
        report_test_status("Image Preprocessing", False)
        return False

# --- Test 2: ONNX Model Loading and Dummy Prediction ---
# This test checks if the ONNX model can be loaded and if it produces an output
# with the correct shape and type when given some random (dummy) input.
def run_onnx_model_test():
    print("\n--- Starting Test: ONNX Model Loading & Dummy Prediction ---")

    # Check if the ONNX model file exists
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"ERROR: ONNX model not found at {ONNX_MODEL_PATH}. Please run 'convert_to_onnx.py' first!")
        report_test_status("ONNX Model Loading", False)
        return False

    try:
        # Create an ONNX Runtime session using create_session from model.py
        print(f"  Attempting to load ONNX model from: {ONNX_MODEL_PATH}")
        session = create_session(ONNX_MODEL_PATH)

        # Create a dummy (fake) input array matching the expected input shape and type
        # Convert to list as model.py's predict function expects a list
        dummy_input_np = np.random.rand(1, 3, 224, 224).astype(np.float32)
        dummy_input_list = dummy_input_np.tolist()

        print("  Sending dummy input (as list) to the model for prediction...")
        output_probabilities = predict(session, dummy_input_list)

        # Check 1: Is the output a NumPy array? (predict returns result[0] which is numpy)
        if not isinstance(output_probabilities, np.ndarray):
            print("  FAIL: Model output is not a NumPy array.")
            report_test_status("ONNX Model Loading & Dummy Prediction", False)
            return False

        # Check 2: Does the output have the correct shape?
        # ImageNet has 1000 classes, so expected shape is (1, 1000)
        expected_output_shape = (1, 1000)
        if output_probabilities.shape != expected_output_shape:
            print(f"  FAIL: Incorrect model output shape. Expected {expected_output_shape}, got {output_probabilities.shape}")
            report_test_status("ONNX Model Loading & Dummy Prediction", False)
            return False

        # Check 3: Is the data type correct (float32)?
        if output_probabilities.dtype != np.float32:
            print(f"  FAIL: Incorrect model output data type. Expected float32, got {output_probabilities.dtype}")
            report_test_status("ONNX Model Loading & Dummy Prediction", False)
            return False

        # Check 4: Are there any problematic values (like 'Not a Number' or 'Infinity')?
        if np.isnan(output_probabilities).any() or np.isinf(output_probabilities).any():
            print("  FAIL: Model output contains NaN or Inf values.")
            report_test_status("ONNX Model Loading & Dummy Prediction", False)
            return False

        report_test_status("ONNX Model Loading & Dummy Prediction", True)
        return True

    except Exception as e:
        print(f"  An unexpected error occurred during ONNX model test: {e}")
        report_test_status("ONNX Model Loading & Dummy Prediction", False)
        return False

# --- Test 3: End-to-End Inference with Real Images ---
# This is the most important test! It combines preprocessing and model prediction
# using actual images and verifies if the model predicts the correct class.
def run_end_to_end_test():
    print("\n--- Starting Test: End-to-End Inference with Real Images ---")
    all_tests_passed = True # Keep track if all sub-tests pass

    # Initialize the ONNX session once
    session = None
    try:
        session = create_session(ONNX_MODEL_PATH)
    except Exception as e:
        print(f"ERROR: Could not create ONNX session: {e}. Cannot run end-to-end tests.")
        report_test_status("End-to-End Inference", False)
        return False # Exit if we can't even load the model

    # Loop through each expected image and its class ID
    for image_name, expected_id in EXPECTED_PREDICTIONS.items():
        image_path = os.path.join(TEST_IMAGES_DIR, image_name)
        print(f"\n  Testing image: {image_name}")

        # Check if the specific image file exists
        if not os.path.exists(image_path):
            print(f"    ERROR: Image not found at {image_path}. Skipping this image test.")
            all_tests_passed = False
            continue # Go to the next image

        try:
            # Step 1: Preprocess the image using preprocess_input from model.py
            print("    Preprocessing image...")
            processed_input_list = preprocess_input(image_path)

            # Step 2: Get prediction from the ONNX model using predict from model.py
            print("    Getting prediction from ONNX model...")
            probabilities = predict(session, processed_input_list)

            # Find the class with the highest probability (this is our prediction)
            predicted_id = np.argmax(probabilities, axis=1)[0]
            confidence = np.max(probabilities) # The probability of the predicted class

            print(f"    Predicted Class ID: {predicted_id}")
            print(f"    Expected Class ID: {expected_id}")
            print(f"    Confidence: {confidence:.4f}") # Print confidence for debugging

            # Check if our prediction matches the expected class ID
            if predicted_id == expected_id:
                print(f"    Result for {image_name}: PASSED")
            else:
                print(f"    Result for {image_name}: FAILED (Predicted {predicted_id}, Expected {expected_id})")
                all_tests_passed = False # Mark that at least one test failed

        except Exception as e:
            # Catch any unexpected errors during the process for this image
            print(f"    An unexpected error occurred for {image_name}: {e}")
            all_tests_passed = False

    report_test_status("End-to-End Inference with Real Images", all_tests_passed)
    return all_tests_passed

# --- Main Test Runner ---
# This function orchestrates all the tests.
def main():
    print("--- Starting All Tests for ML Model Deployment ---")
    overall_success = True # Tracks if *all* tests pass

    # First, check if the images directory exists
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"ERROR: The '{TEST_IMAGES_DIR}' directory is missing. Please create it and put your test images inside.")
        print("Tests cannot run without the images.")
        overall_success = False # Mark overall failure

    if overall_success: # Only run tests if the image directory and model file exist
        # Run each test function and update the overall success status
        overall_success &= run_preprocessing_test()
        overall_success &= run_onnx_model_test()
        overall_success &= run_end_to_end_test()

    print("\n--- All Tests Completed ---")
    if overall_success:
        print("🥳 Overall Test Status: ALL TESTS PASSED! Your model seems ready for deployment.")
    else:
        print("😔 Overall Test Status: SOME TESTS FAILED. Please review the failures above.")
        # Optionally, uncomment the line below to exit with an error code on failure.
        # import sys; sys.exit(1)

if __name__ == "__main__":
    main()