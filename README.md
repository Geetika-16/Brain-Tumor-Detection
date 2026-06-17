# 🧠 AI-Based Brain Tumor Detection and Analysis System

## Description

AI-Based Brain Tumor Detection and Analysis System is a hospital-oriented web application designed to assist healthcare professionals in detecting and analyzing brain tumors from MRI scans. The system integrates Deep Learning, Computer Vision, Explainable AI, and Hospital Management functionalities into a unified platform.

The application enables administrators to manage doctors and patients, doctors to perform AI-assisted MRI analysis, and patients to access their reports through a secure portal. A YOLO model is used to detect and localize tumor regions, while an EfficientNet-based CNN classifies the detected tumor into one of four categories: Glioma, Meningioma, Pituitary Tumor, or No Tumor.

To improve transparency and trust in AI predictions, Grad-CAM visualizations highlight the MRI regions responsible for the classification. The system also performs feature extraction and uncertainty analysis to provide additional clinical insights and improve decision-making.

---

## Features

### 🏥 Hospital Management

* Admin, Doctor, and Patient modules.
* Patient allocation to available doctors.
* MRI report management through a centralized database.
* Secure login system for doctors and patients.

### 🌐 Multilingual Support

* Patient interface translation using Google Translator.
* Improved accessibility for users from different language backgrounds.

### 🧠 Brain Tumor Detection

* Automatic tumor localization using YOLO.
* Bounding box generation around detected tumor regions.
* Tumor detection confidence score.

### 🔬 Brain Tumor Classification

* EfficientNet-based CNN classification model.
* Classification into:

  * Glioma
  * Meningioma
  * Pituitary Tumor
  * No Tumor
* Classification confidence score generation.

### 📊 Explainable AI (XAI)

* Grad-CAM heatmap visualization.
* Explanation of CNN predictions.
* Visualization on both cropped tumor regions and full MRI images.

### 📈 Feature Extraction

* Tumor Intensity Analysis
* Texture Analysis
* Tumor Area Calculation
* Tumor Perimeter Measurement
* Circularity Measurement
* Tumor Percentage Estimation
* Bounding Box Dimension Analysis
* Tumor Location Identification

### ⚠️ Uncertainty Analysis

* Comparison of YOLO and CNN confidence scores.
* Reliability assessment of AI predictions.
* Alert generation for uncertain cases requiring additional review.

### 📄 Patient Report System

* Doctor-generated diagnostic reports.
* Patient report viewing portal.
* Database-driven report storage and retrieval.

---

## Technologies Used

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Flask (Python)

### Database

* MySQL

### Deep Learning & Computer Vision

* YOLO
* EfficientNet
* Convolutional Neural Network (CNN)
* TensorFlow
* OpenCV

### Explainable AI

* Grad-CAM

### Data Processing

* Feature Extraction
* Uncertainty Analysis

### Additional Tools

* Google Translator API
* NumPy
* Pandas

---

## Project Workflow

### Step 1: Patient and Doctor Management

1. Admin manages doctor and patient records.
2. MRI scans are uploaded and stored in the database.
3. Admin assigns patients to available doctors.
4. Doctors access assigned patient records.

### Step 2: MRI Retrieval

1. Doctor selects a patient from the dashboard.
2. MRI scan is retrieved automatically from the database.
3. Doctor initiates AI analysis.

### Step 3: Tumor Detection using YOLO

1. MRI image is processed by the YOLO model.
2. Tumor location is detected.
3. Bounding box is generated around the tumor region.
4. Detection confidence score is produced.

### Step 4: Tumor Region Cropping

1. Detected tumor region is automatically cropped.
2. Cropped tumor image is prepared for classification.

### Step 5: Tumor Classification using CNN

1. Cropped tumor image is passed to the EfficientNet-based CNN model.
2. CNN predicts the tumor type.
3. Classification confidence score is generated.
4. Possible outputs:

   * Glioma
   * Meningioma
   * Pituitary Tumor
   * No Tumor

### Step 6: Explainable AI Analysis

1. Grad-CAM generates heatmaps.
2. Important regions influencing the CNN prediction are highlighted.
3. Visual explanations are provided for both cropped and full MRI images.

### Step 7: Feature Extraction

1. Tumor intensity features are extracted.
2. Texture features are analyzed.
3. Shape features such as area, perimeter, and circularity are calculated.
4. Tumor percentage and location are determined.
5. Bounding box dimensions are measured.

### Step 8: Uncertainty Analysis

1. YOLO and CNN confidence scores are evaluated.
2. Prediction reliability is assessed.
3. Uncertain cases are flagged for additional medical review.

### Step 9: Report Generation

1. AI analysis results are compiled into a report.
2. Doctor reviews and validates the findings.
3. Final report is stored in the database.

### Step 10: Patient Access

1. Patient logs into the portal.
2. Reports and doctor messages are retrieved.
3. Content can be translated into the patient's preferred language.
4. Patient reviews results and follows up with the doctor if required.

---

### Files Explanation
1. Main.py - Main code of the program where integrates all the steps together and sent to flask.
2. app.py - Backend flask code to integrate the web templates and project flow together.
3. Templates - Web designing Files

---

## Outcome

The system provides an AI-assisted, explainable, and hospital-integrated solution for brain tumor detection and analysis. By combining YOLO-based localization, CNN-based classification, Grad-CAM explainability, feature extraction, and uncertainty analysis, the platform supports healthcare professionals in making more informed diagnostic decisions while improving patient accessibility and report management.

