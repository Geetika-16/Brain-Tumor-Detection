import os
from datetime import datetime

from flask import Flask, render_template, request, url_for, send_from_directory, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import mysql.connector

from main import process_image

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "final_pipeline")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Geeti#2216",
        database="neuroai"
    )


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def to_web_path(path_value):
    if not path_value:
        return None
    rel_path = os.path.relpath(path_value, BASE_DIR).replace("\\", "/")
    return url_for("result_files", filename=rel_path)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/result-files/<path:filename>")
def result_files(filename):
    return send_from_directory(BASE_DIR, filename)


# ---------------- PAGE ROUTES ----------------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/admin")
def admin():
    return render_template("admin.html")


@app.route("/doctor")
def doctor():
    return render_template("doctor.html")


@app.route("/patient")
def patient():
    return render_template("patient.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    patient_id = request.args.get("patient_id")
    report_id = request.args.get("report_id")

    result = None
    error = None
    patient = None
    patient_mri_url = None

    if not patient_id or not report_id:
        return render_template("index.html", error="Missing patient_id or report_id")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT
            p.patient_id,
            p.patient_code,
            p.patient_name,
            p.age,
            p.gender,
            p.phone_no,
            DATE_FORMAT(p.admission_date, '%d %b %Y') AS admission_date,
            m.report_id,
            m.mri_image
        FROM patients p
        JOIN mri_reports m ON p.patient_id = m.patient_id
        WHERE p.patient_id = %s AND m.report_id = %s
    """, (patient_id, report_id))

    patient = cursor.fetchone()

    if not patient:
        cursor.close()
        conn.close()
        return render_template("index.html", error="Patient MRI report not found")

    mri_db_path = patient["mri_image"]  # example: uploads/patient_1_mri.jpg
    mri_file_path = os.path.join(BASE_DIR, mri_db_path.replace("/", os.sep))
    patient_mri_url = "/" + mri_db_path.replace("\\", "/")

    if request.method == "POST":
        try:
            result = process_image(mri_file_path)

            if result and result.get("gradcam_files"):
                for key, value in result["gradcam_files"].items():
                    result["gradcam_files"][key] = to_web_path(value)

            tumor_type = result.get("tumor_type")
            tumor_location = result.get("features", {}).get("location") if result.get("features") else None
            confidence_score = None

            if result.get("yolo_confidence") is not None:
                confidence_score = round(float(result.get("yolo_confidence")) * 100, 2)

            clinical_report = result.get("ai_explanation") or result.get("reason")

            cursor.execute("""
                UPDATE mri_reports
                SET
                    prediction_status = 'Completed',
                    tumor_detected = %s,
                    tumor_type = %s,
                    tumor_location = %s,
                    confidence_score = %s,
                    clinical_report = %s
                WHERE report_id = %s
            """, (
                "Yes" if tumor_type and tumor_type.lower() != "no tumor" else "No",
                tumor_type,
                tumor_location,
                confidence_score,
                clinical_report,
                report_id
            ))

            conn.commit()

        except Exception as e:
            error = f"Pipeline error: {str(e)}"

    cursor.close()
    conn.close()

    return render_template(
        "index.html",
        result=result,
        error=error,
        patient=patient,
        patient_mri_url=patient_mri_url
    )


# ---------------- ADMIN APIs ----------------
@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT id, username, email
        FROM admin
        WHERE email = %s AND password = %s
    """, (email, password))

    admin = cursor.fetchone()
    cursor.close()
    conn.close()

    if admin:
        return jsonify({"success": True, "admin": admin})
    return jsonify({"success": False, "message": "Invalid admin credentials"}), 401

@app.route("/api/admin/patients", methods=["GET"])
def get_admin_patients():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT 
            p.patient_id,
            p.patient_code,
            p.patient_name,
            p.email,
            p.age,
            p.gender,
            p.phone_no,
            p.address,
            DATE_FORMAT(p.admission_date, '%d %b %Y') AS admission_date,
            p.doctor_id,
            p.status,

            d.Doc_name AS doctor_name,
            d.specialization AS doctor_specialization,

            m.report_id,
            m.mri_image,
            DATE_FORMAT(m.scan_date, '%d %b %Y') AS scan_date,
            m.mri_status,
            m.prediction_status,
            m.tumor_detected,
            m.tumor_type,
            m.tumor_location,
            m.confidence_score,
            m.severity,
            m.report_sent
        FROM patients p
        LEFT JOIN doctors d ON p.doctor_id = d.Doc_id
        LEFT JOIN mri_reports m ON p.patient_id = m.patient_id
        ORDER BY p.patient_id ASC
    """)

    patients = cursor.fetchall()
    cursor.close()
    conn.close()

    return jsonify(patients)


@app.route("/api/admin/doctors", methods=["GET"])
def get_admin_doctors():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT 
            Doc_id AS doctor_id,
            Doc_name AS doctor_name,
            email,
            specialization,
            Contact_no AS contact_no,
            experience,
            duty_time,
            Status AS status
        FROM doctors
        ORDER BY Doc_id ASC
    """)

    doctors = cursor.fetchall()
    cursor.close()
    conn.close()

    return jsonify(doctors)


@app.route("/api/admin/dashboard", methods=["GET"])
def get_admin_dashboard():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT COUNT(*) AS total_patients FROM patients")
    total_patients = cursor.fetchone()["total_patients"]

    cursor.execute("SELECT COUNT(*) AS total_doctors FROM doctors")
    total_doctors = cursor.fetchone()["total_doctors"]

    cursor.execute("SELECT COUNT(*) AS total_scans FROM mri_reports WHERE mri_status = 'Uploaded'")
    total_scans = cursor.fetchone()["total_scans"]

    cursor.execute("SELECT COUNT(*) AS reports_generated FROM mri_reports WHERE report_sent = 'Yes'")
    reports_generated = cursor.fetchone()["reports_generated"]

    cursor.execute("SELECT COUNT(*) AS unallocated FROM patients WHERE doctor_id IS NULL")
    unallocated = cursor.fetchone()["unallocated"]

    cursor.execute("SELECT COUNT(*) AS pending_predictions FROM mri_reports WHERE prediction_status = 'Pending'")
    pending_predictions = cursor.fetchone()["pending_predictions"]

    cursor.close()
    conn.close()

    return jsonify({
        "total_patients": total_patients,
        "total_doctors": total_doctors,
        "total_scans": total_scans,
        "reports_generated": reports_generated,
        "unallocated": unallocated,
        "pending_predictions": pending_predictions
    })


@app.route("/api/admin/assign-doctor", methods=["POST"])
def assign_doctor():
    data = request.get_json()

    patient_id = data.get("patient_id")
    doctor_id = data.get("doctor_id")

    if not patient_id or not doctor_id:
        return jsonify({"success": False, "message": "patient_id and doctor_id are required"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE patients
        SET doctor_id = %s,
            status = 'Assigned'
        WHERE patient_id = %s
    """, (doctor_id, patient_id))

    cursor.execute("""
        UPDATE mri_reports
        SET doctor_id = %s
        WHERE patient_id = %s
    """, (doctor_id, patient_id))

    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"success": True, "message": "Doctor assigned successfully"})

@app.route("/api/doctor/login", methods=["POST"])
def doctor_login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT 
            Doc_id AS doctor_id,
            Doc_name AS doctor_name,
            email,
            specialization,
            Contact_no AS contact_no,
            experience,
            duty_time,
            Status AS status
        FROM doctors
        WHERE email = %s AND password = %s
    """, (email, password))

    doctor = cursor.fetchone()

    cursor.close()
    conn.close()

    if doctor:
        return jsonify({"success": True, "doctor": doctor})

    return jsonify({"success": False, "message": "Invalid doctor login"}), 401

@app.route("/api/doctor/patients/<int:doctor_id>", methods=["GET"])
def get_doctor_patients(doctor_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT
            p.patient_id,
            p.patient_code,
            p.patient_name,
            p.age,
            p.gender,
            p.phone_no,
            DATE_FORMAT(p.admission_date, '%d %b %Y') AS admission_date,
            p.status,

            m.report_id,
            m.mri_image,
            m.mri_status,
            m.prediction_status,
            m.tumor_detected,
            m.tumor_type,
            m.tumor_location,
            m.confidence_score,
            m.severity,
            m.clinical_report,
            m.doctor_message,
            m.report_sent
        FROM patients p
        LEFT JOIN mri_reports m ON p.patient_id = m.patient_id
        WHERE p.doctor_id = %s
        ORDER BY p.patient_id ASC
    """, (doctor_id,))

    patients = cursor.fetchall()

    cursor.close()
    conn.close()

    return jsonify(patients)

@app.route("/api/doctor/send-report", methods=["POST"])
def send_report():
    data = request.get_json()

    report_id = data.get("report_id")
    tumor_type = data.get("tumor_type")
    clinical_report = data.get("clinical_report")
    doctor_message = data.get("doctor_message")

    if not report_id:
        return jsonify({
            "success": False,
            "message": "report_id is required"
        }), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE mri_reports
        SET
            tumor_type = %s,
            clinical_report = %s,
            doctor_message = %s,
            report_sent = 'Yes'
        WHERE report_id = %s
    """, (tumor_type, clinical_report, doctor_message, report_id))

    conn.commit()

    cursor.close()
    conn.close()

    return jsonify({
        "success": True,
        "message": "Report sent successfully"
    })

@app.route("/api/patient/login", methods=["POST"])
def patient_login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT 
            patient_id,
            patient_code,
            patient_name,
            email,
            age,
            gender,
            phone_no,
            address,
            DATE_FORMAT(admission_date, '%d %b %Y') AS admission_date,
            doctor_id,
            status
        FROM patients
        WHERE email = %s AND password = %s
    """, (email, password))

    patient = cursor.fetchone()

    cursor.close()
    conn.close()

    if patient:
        return jsonify({"success": True, "patient": patient})

    return jsonify({"success": False, "message": "Invalid patient login"}), 401

@app.route("/api/patient/report/<int:patient_id>", methods=["GET"])
def patient_report(patient_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT
            p.patient_id,
            p.patient_code,
            p.patient_name,
            p.email,
            p.age,
            p.gender,
            p.phone_no,
            p.address,
            DATE_FORMAT(p.admission_date, '%d %b %Y') AS admission_date,

            d.Doc_id AS doctor_id,
            d.Doc_name AS doctor_name,
            d.email AS doctor_email,
            d.specialization,
            d.Contact_no AS doctor_phone,
            d.experience,
            d.duty_time,
            d.Status AS doctor_status,

            m.report_id,
            DATE_FORMAT(m.scan_date, '%d %b %Y') AS scan_date,
            m.mri_status,
            m.prediction_status,
            m.tumor_detected,
            m.tumor_type,
            m.tumor_location,
            m.confidence_score,
            m.severity,
            m.clinical_report,
            m.doctor_message,
            m.report_sent,
            DATE_FORMAT(m.created_at, '%d %b %Y, %H:%i') AS report_date
        FROM patients p
        LEFT JOIN doctors d ON p.doctor_id = d.Doc_id
        LEFT JOIN mri_reports m ON p.patient_id = m.patient_id
        WHERE p.patient_id = %s
        ORDER BY m.report_id DESC
        LIMIT 1
    """, (patient_id,))

    report = cursor.fetchone()

    cursor.close()
    conn.close()

    if report:
        return jsonify({"success": True, "data": report})

    return jsonify({"success": False, "message": "Report not found"}), 404

@app.route("/api/patient/register", methods=["POST"])
def patient_register():
    data = request.get_json()

    patient_name = data.get("patient_name")
    email = data.get("email")
    password = data.get("password")
    phone_no = data.get("phone_no")
    age = data.get("age")
    gender = data.get("gender")
    address = data.get("address")

    if not patient_name or not email or not password:
        return jsonify({
            "success": False,
            "message": "Name, email and password are required"
        }), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT patient_id FROM patients WHERE email = %s", (email,))
    existing = cursor.fetchone()

    if existing:
        cursor.close()
        conn.close()
        return jsonify({
            "success": False,
            "message": "Email already registered"
        }), 409

    cursor.execute("SELECT COUNT(*) AS total FROM patients")
    total = cursor.fetchone()["total"]
    next_code = f"P-{total + 1:03d}"

    cursor.execute("""
        INSERT INTO patients
        (patient_code, patient_name, email, password, age, gender, phone_no, address, admission_date, doctor_id, status)
        VALUES
        (%s, %s, %s, %s, %s, %s, %s, %s, CURDATE(), NULL, 'Pending')
    """, (
        next_code,
        patient_name,
        email,
        password,
        age,
        gender,
        phone_no,
        address
    ))

    conn.commit()
    patient_id = cursor.lastrowid

    cursor.close()
    conn.close()

    return jsonify({
        "success": True,
        "message": "Registration successful. Please login.",
        "patient_id": patient_id,
        "patient_code": next_code
    })

if __name__ == "__main__":
    app.run(debug=True)
