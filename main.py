import os
import json
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model

YOLO_MODEL_PATH = "models/yolo/runs/detect/brain_tumor_yolo-2/weights/best.pt"
CNN_MODEL_DIR = "models/cnn_3class"
CNN_WEIGHTS_PATH = os.path.join(CNN_MODEL_DIR, "final_model.weights.h5")
CNN_METADATA_PATH = os.path.join(CNN_MODEL_DIR, "metadata.json")
OUTPUT_DIR = "outputs/final_pipeline4"

os.makedirs(OUTPUT_DIR, exist_ok=True)

YOLO_CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
YOLO_CONF_THRESHOLD = 0.40
CNN_CONF_THRESHOLD = 0.60
LAST_CONV_LAYER_NAME = "top_conv"

#Using metadata to avoid hardcoded data and makes dynamic 
with open(CNN_METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f) #loads cnn details

CLASS_NAMES = metadata["class_names"]
IMG_SIZE = tuple(metadata["img_size"])
NUM_CLASSES = metadata["num_classes"]

#CNN model building
def build_cnn_model(num_classes, img_size):
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size[0], img_size[1], 3),
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x) #Converts deep feature maps into compact fea vector+
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x) #learns tumor patterns
    x = Dropout(0.35)(x) #prevents overfitting
    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=base_model.input, outputs=outputs)


yolo_model = YOLO(YOLO_MODEL_PATH)
cnn_model = build_cnn_model(NUM_CLASSES, IMG_SIZE)
cnn_model.load_weights(CNN_WEIGHTS_PATH)


def empty_gradcam_files():
    return {
        "bbox": None,
        "gradcam": None,
        "overlay": None,
        "cropped": None,
        "cropped_gradcam": None,
        "cropped_overlay": None,
        "crop_image": None,
        "heatmap_image": None,
        "overlay_image": None,
        "boxed_image": None,
        "full_heatmap_image": None,
        "full_overlay_image": None,
    }


def expand_box(x1, y1, x2, y2, img_w, img_h, scale=0.20):
    box_w = x2 - x1
    box_h = y2 - y1

    pad_w = int(box_w * scale)
    pad_h = int(box_h * scale)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(img_w, x2 + pad_w)
    y2 = min(img_h, y2 + pad_h)

    return x1, y1, x2, y2

#converts image to RGB
def classify_crop(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    batch = np.expand_dims(resized.astype("float32"), axis=0)

    pred = cnn_model.predict(batch, verbose=0)[0]
    class_id = int(np.argmax(pred))
    confidence = float(np.max(pred))

    return CLASS_NAMES[class_id], confidence, pred, batch


def classify_full_image(original_bgr):
    rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    batch = np.expand_dims(resized.astype("float32"), axis=0)
    return batch


def get_uncertainty(yolo_class, yolo_conf, cnn_class=None, cnn_conf=None):
    if yolo_class == "No Tumor":
        if yolo_conf >= 0.70:
            return "Low", "YOLO predicted no tumor with strong confidence."
        return "Medium", "YOLO predicted no tumor with moderate confidence."

    if cnn_class is None:
        return "High", "CNN result unavailable."

    if yolo_class != cnn_class:
        return "High", "YOLO and CNN predictions do not match."

    if yolo_conf >= 0.75 and cnn_conf >= 0.75:
        return "Low", "YOLO and CNN match with high confidence."

    if yolo_conf >= YOLO_CONF_THRESHOLD and cnn_conf >= CNN_CONF_THRESHOLD:
        return "Medium", "YOLO and CNN match, but confidence is moderate."

    return "High", "Predictions match, but confidence is low."


def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_image(original_rgb, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_rgb.shape[1], original_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_rgb, 1 - alpha, heatmap_color_rgb, alpha, 0)
    return heatmap_color, overlay


def get_binary_mask(gray_img): #seperates tumor region from background
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_intensity_features(gray_img):
    return {
        "mean_intensity": round(float(np.mean(gray_img)), 4),
        "std_intensity": round(float(np.std(gray_img)), 4),
    }


def extract_texture_features(gray_img):
    glcm = graycomatrix(
        gray_img,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )

    return {
        "texture_contrast": round(float(graycoprops(glcm, "contrast")[0, 0]), 4),
        "texture_energy": round(float(graycoprops(glcm, "energy")[0, 0]), 4),
    }


def extract_shape_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {
            "tumor_area": 0.0,
            "tumor_perimeter": 0.0,
            "circularity": 0.0,
        }

    largest_contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest_contour))
    perimeter = float(cv2.arcLength(largest_contour, True))

    circularity = 0.0
    if perimeter > 0:
        circularity = float((4 * np.pi * area) / (perimeter ** 2))

    return {
        "tumor_area": round(area, 4),
        "tumor_perimeter": round(perimeter, 4),
        "circularity": round(circularity, 4),
    }


def extract_spatial_features(box, full_img_shape, mask):
    x1, y1, x2, y2 = box
    full_h, full_w = full_img_shape[:2]

    box_width = x2 - x1
    box_height = y2 - y1
    box_area = box_width * box_height

    mask_area = float(np.count_nonzero(mask))
    total_image_area = float(full_h * full_w)
    tumor_area_percent = (mask_area / total_image_area) * 100 if total_image_area > 0 else 0.0

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    horizontal = "Left" if center_x < (full_w / 2) else "Right"
    vertical = "Upper" if center_y < (full_h / 2) else "Lower"

    return {
        "bounding_box_width": int(box_width),
        "bounding_box_height": int(box_height),
        "bounding_box_area": int(box_area),
        "tumor_area_percent": round(tumor_area_percent, 4),
        "location": f"{horizontal} {vertical}",
    }


def extract_all_features(crop_img, box, full_img_shape):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    mask = get_binary_mask(gray)

    features = {}
    features.update(extract_intensity_features(gray))
    features.update(extract_texture_features(gray))
    features.update(extract_shape_features(mask))
    features.update(extract_spatial_features(box, full_img_shape, mask))

    return features

#Converts extracted features into readable medical explanation
def build_ai_explanation(final_type, uncertainty, agreement, features):
    intensity = features["mean_intensity"]
    contrast = features["texture_contrast"]
    circularity = features["circularity"]
    location = features["location"]

    shape_text = "irregular" if circularity < 0.5 else "moderately regular"
    intensity_text = "high intensity" if intensity > 100 else "moderate intensity"
    texture_text = "high texture contrast" if contrast > 2.0 else "mild texture contrast"

    if final_type == "Uncertain":
        return (
            f"The models produced mismatched predictions, so the result is uncertain. "
            f"The detected region is in the {location} area and shows {intensity_text}, "
            f"{texture_text}, and a {shape_text} shape. Doctor review is recommended."
        )

    return (
        f"The system identified a {final_type} tumor. The detected region is in the {location} area. "
        f"Feature analysis shows {intensity_text}, {texture_text}, and a {shape_text} shape. "
        f"Model agreement status is {agreement.lower()} with {uncertainty.lower()} uncertainty."
    )


def save_report(file_name, report):
    report_path = os.path.join(OUTPUT_DIR, f"{file_name}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n===== FINAL REPORT =====")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report: {report_path}")

    return report

#controls entire AI pipeline
def process_image(image_path):
    img = cv2.imread(image_path) #loads mri img

    if img is None:
        return {
            "tumor_type": "Error",
            "yolo_prediction": None,
            "yolo_confidence": None,
            "cnn_prediction": None,
            "cnn_confidence": None,
            "agreement_status": "Not Available",
            "uncertainty": "High",
            "reason": "Image not found.",
            "features": None,
            "gradcam_files": empty_gradcam_files(),
            "ai_explanation": "The input image could not be loaded.",
        }

    original = img.copy()
    file_name = os.path.splitext(os.path.basename(image_path))[0]

    results = yolo_model(img)[0] #detects tumor region

    if results.boxes is None or len(results.boxes) == 0:
        report = {
            "tumor_type": "No Tumor",
            "yolo_prediction": "No Tumor",
            "yolo_confidence": 1.0,
            "cnn_prediction": None,
            "cnn_confidence": None,
            "agreement_status": "Not Applicable",
            "uncertainty": "Low",
            "reason": "YOLO found no tumor region.",
            "features": None,
            "gradcam_files": empty_gradcam_files(),
            "ai_explanation": "No tumor region was detected in the MRI image.",
        }
        return save_report(file_name, report)

    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)

    best_idx = int(np.argmax(scores))
    yolo_conf = float(scores[best_idx])
    yolo_class_name = YOLO_CLASSES[int(class_ids[best_idx])]

    if yolo_conf < YOLO_CONF_THRESHOLD:
        report = {
            "tumor_type": "Uncertain",
            "yolo_prediction": yolo_class_name,
            "yolo_confidence": round(yolo_conf, 4),
            "cnn_prediction": None,
            "cnn_confidence": None,
            "agreement_status": "Not Available",
            "uncertainty": "High",
            "reason": "YOLO confidence is too low.",
            "features": None,
            "gradcam_files": empty_gradcam_files(),
            "ai_explanation": "The system detected a possible tumor region, but confidence is too low for a reliable decision.",
        }
        return save_report(file_name, report)

    if yolo_class_name == "No Tumor":
        uncertainty, reason = get_uncertainty(yolo_class_name, yolo_conf)

        report = {
            "tumor_type": "No Tumor",
            "yolo_prediction": yolo_class_name,
            "yolo_confidence": round(yolo_conf, 4),
            "cnn_prediction": None,
            "cnn_confidence": None,
            "agreement_status": "Not Applicable",
            "uncertainty": uncertainty,
            "reason": reason,
            "features": None,
            "gradcam_files": empty_gradcam_files(),
            "ai_explanation": "YOLO predicted no tumor, so CNN, Grad-CAM, and tumor feature extraction were skipped.",
        }
        return save_report(file_name, report)

    x1, y1, x2, y2 = map(int, boxes[best_idx])
    h, w = original.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        report = {
            "tumor_type": "Uncertain",
            "yolo_prediction": yolo_class_name,
            "yolo_confidence": round(yolo_conf, 4),
            "cnn_prediction": None,
            "cnn_confidence": None,
            "agreement_status": "Not Available",
            "uncertainty": "High",
            "reason": "Invalid YOLO bounding box.",
            "features": None,
            "gradcam_files": empty_gradcam_files(),
            "ai_explanation": "The detected region could not be processed correctly.",
        }
        return save_report(file_name, report)

    x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h, scale=0.20)
    crop = original[y1:y2, x1:x2]

    if crop.size == 0:
        report = {
            "tumor_type": "Uncertain",
            "yolo_prediction": yolo_class_name,
            "yolo_confidence": round(yolo_conf, 4),
            "cnn_prediction": None,
            "cnn_confidence": None,
            "agreement_status": "Not Available",
            "uncertainty": "High",
            "reason": "Empty crop after detection.",
            "features": None,
            "gradcam_files": empty_gradcam_files(),
            "ai_explanation": "The detected tumor crop could not be extracted correctly.",
        }
        return save_report(file_name, report)

    cnn_class_name, cnn_conf, probs, cnn_input = classify_crop(crop)

    if yolo_class_name == cnn_class_name:
        final_type = cnn_class_name
        agreement = "Matched"
    else:
        final_type = "Uncertain"
        agreement = "Mismatched"

    uncertainty, reason = get_uncertainty(yolo_class_name, yolo_conf, cnn_class_name, cnn_conf)

    crop_heatmap = get_gradcam_heatmap(cnn_input, cnn_model, LAST_CONV_LAYER_NAME)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_heatmap_color, crop_overlay = overlay_heatmap_on_image(crop_rgb, crop_heatmap)

    full_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    full_batch = classify_full_image(original)
    full_heatmap = get_gradcam_heatmap(full_batch, cnn_model, LAST_CONV_LAYER_NAME)
    full_heatmap_color, full_overlay = overlay_heatmap_on_image(full_rgb, full_heatmap)

    features = extract_all_features(crop, (x1, y1, x2, y2), original.shape)
    ai_explanation = build_ai_explanation(final_type, uncertainty, agreement, features)

    crop_path = os.path.join(OUTPUT_DIR, f"{file_name}_crop.png")
    crop_heatmap_path = os.path.join(OUTPUT_DIR, f"{file_name}_heatmap.png")
    crop_overlay_path = os.path.join(OUTPUT_DIR, f"{file_name}_overlay.png")
    boxed_path = os.path.join(OUTPUT_DIR, f"{file_name}_boxed.png")
    full_heatmap_path = os.path.join(OUTPUT_DIR, f"{file_name}_full_heatmap.png")
    full_overlay_path = os.path.join(OUTPUT_DIR, f"{file_name}_full_overlay.png")

    boxed_img = original.copy()
    cv2.rectangle(boxed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        boxed_img,
        f"{final_type}",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.imwrite(crop_path, crop)
    cv2.imwrite(crop_heatmap_path, crop_heatmap_color)
    cv2.imwrite(crop_overlay_path, cv2.cvtColor(crop_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(boxed_path, boxed_img)
    cv2.imwrite(full_heatmap_path, full_heatmap_color)
    cv2.imwrite(full_overlay_path, cv2.cvtColor(full_overlay, cv2.COLOR_RGB2BGR))

    report = {
        "tumor_type": final_type,
        "yolo_prediction": yolo_class_name,
        "yolo_confidence": round(yolo_conf, 4),
        "cnn_prediction": cnn_class_name,
        "cnn_confidence": round(cnn_conf, 4),
        "agreement_status": agreement,
        "uncertainty": uncertainty,
        "reason": reason,
        "bounding_box": [x1, y1, x2, y2],
        "class_probabilities": {
            name: round(float(score), 4) for name, score in zip(CLASS_NAMES, probs)
        },
        "features": features,
        "gradcam_files": {
            "bbox": boxed_path,
            "gradcam": full_heatmap_path,
            "overlay": full_overlay_path,
            "cropped": crop_path,
            "cropped_gradcam": crop_heatmap_path,
            "cropped_overlay": crop_overlay_path,
            "crop_image": crop_path,
            "heatmap_image": crop_heatmap_path,
            "overlay_image": crop_overlay_path,
            "boxed_image": boxed_path,
            "full_heatmap_image": full_heatmap_path,
            "full_overlay_image": full_overlay_path,
        },
        "ai_explanation": ai_explanation,
    }

    return save_report(file_name, report)


if __name__ == "__main__":
    while True:
        path = input("\nEnter MRI image path (or exit): ").strip()
        if path.lower() == "exit":
            break
        process_image(path)
