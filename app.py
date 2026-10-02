from flask import Flask, render_template, request, send_from_directory, send_file
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import numpy as np
import joblib


# ======================================================
# FLASK APPLICATION
# ======================================================

app = Flask(__name__)


# ======================================================
# LOAD TRAINED MODELS
# ======================================================

diabetes_model = joblib.load("models/diabetes_model.pkl")
diabetes_scaler = joblib.load("models/scaler.pkl")

heart_model = joblib.load("models/heart_model.pkl")
heart_scaler = joblib.load("models/heart_scaler.pkl")

cancer_model = joblib.load("models/cancer_model.pkl")
cancer_scaler = joblib.load("models/cancer_scaler.pkl")


# ======================================================
# STORE RECENT PREDICTIONS
# ======================================================

prediction_history = []

# Stores the latest prediction for PDF report
latest_report = {}


# ======================================================
# HELPER FUNCTION
# ======================================================

def save_prediction(module, patient_name, result, confidence, advice):
    """
    Save the latest prediction and also add it to
    the recent prediction history.
    """

    global latest_report

    prediction_time = datetime.now().strftime("%d-%m-%Y %I:%M %p")

    confidence_text = f"{confidence:.2f}%"

    # -----------------------------------------------
    # Save latest report
    # -----------------------------------------------

    latest_report = {
        "module": module,
        "patient": patient_name,
        "result": result,
        "confidence": confidence_text,
        "time": prediction_time,
        "advice": advice
    }

    # -----------------------------------------------
    # Add to prediction history
    # -----------------------------------------------

    prediction_history.insert(
        0,
        {
            "time": prediction_time,
            "module": module,
            "patient": patient_name,
            "result": result,
            "confidence": confidence_text
        }
    )

    # Keep only latest 5 predictions
    prediction_history[:] = prediction_history[:5]


# ======================================================
# HOME PAGE
# ======================================================

@app.route("/")
def home():

    model_accuracy = {
        "Diabetes": 78,
        "Heart Disease": 84,
        "Breast Cancer": 98
    }

    return render_template(
        "index.html",
        history=prediction_history,
        accuracy=model_accuracy
    )


# ======================================================
# FAVICON
# ======================================================

@app.route("/favicon.ico")
def favicon():

    return send_from_directory(
        "static/images",
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon"
    )


# ======================================================
# DOWNLOAD PDF REPORT
# ======================================================

@app.route("/download_report")
def download_report():

    # -----------------------------------------------
    # Check whether a prediction exists
    # -----------------------------------------------

    if not latest_report:

        return (
            "No prediction report available. "
            "Please make a prediction first."
        )

    # -----------------------------------------------
    # Create PDF in memory
    # -----------------------------------------------

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    story = []

    # -----------------------------------------------
    # Report Title
    # -----------------------------------------------

    story.append(
        Paragraph(
            "<b>AI Healthcare Prediction Report</b>",
            styles["Title"]
        )
    )

    story.append(
        Paragraph(
            "<br/>",
            styles["Normal"]
        )
    )

    # -----------------------------------------------
    # Patient Information
    # -----------------------------------------------

    story.append(
        Paragraph(
            f"<b>Patient Name:</b> "
            f"{latest_report.get('patient', 'Patient')}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Module:</b> "
            f"{latest_report.get('module', '-')}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Result:</b> "
            f"{latest_report.get('result', '-')}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Confidence:</b> "
            f"{latest_report.get('confidence', '-')}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Date &amp; Time:</b> "
            f"{latest_report.get('time', '-')}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            "<br/>",
            styles["Normal"]
        )
    )

    # -----------------------------------------------
    # Health Suggestions
    # -----------------------------------------------

    story.append(
        Paragraph(
            "<b>Health Suggestions</b>",
            styles["Heading2"]
        )
    )

    for item in latest_report.get("advice", []):

        story.append(
            Paragraph(
                f"• {item}",
                styles["Normal"]
            )
        )

    story.append(
        Paragraph(
            "<br/>",
            styles["Normal"]
        )
    )

    # -----------------------------------------------
    # Disclaimer
    # -----------------------------------------------

    story.append(
        Paragraph(
            "This report is generated for educational purposes only "
            "and does not replace professional medical advice or diagnosis.",
            styles["Italic"]
        )
    )

    # -----------------------------------------------
    # Generate PDF
    # -----------------------------------------------

    doc.build(story)

    buffer.seek(0)

    # -----------------------------------------------
    # Create module-specific filename
    # -----------------------------------------------

    module = latest_report.get("module", "Healthcare")

    safe_module = (
        module
        .replace(" ", "_")
        .replace("/", "_")
    )

    filename = f"AI_Healthcare_{safe_module}_Report.pdf"

    # -----------------------------------------------
    # Terminal information
    # -----------------------------------------------

    print("======================================")
    print("PDF REPORT GENERATED")
    print("Module:", latest_report.get("module"))
    print("Patient:", latest_report.get("patient"))
    print("Result:", latest_report.get("result"))
    print("Confidence:", latest_report.get("confidence"))
    print("Time:", latest_report.get("time"))
    print("======================================")

    return send_file(
        buffer,
        as_attachment=True,
        download_name=filename,
        mimetype="application/pdf"
    )


# ======================================================
# DIABETES MODULE
# ======================================================

@app.route("/diabetes")
def diabetes():

    return render_template("diabetes.html")


@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():

    try:

        # -------------------------------------------
        # Patient name
        # -------------------------------------------

        patient_name = request.form.get(
            "patient_name",
            "Patient"
        )

        # -------------------------------------------
        # Diabetes features
        # -------------------------------------------

        features = [
            float(request.form["pregnancies"]),
            float(request.form["glucose"]),
            float(request.form["blood_pressure"]),
            float(request.form["skin_thickness"]),
            float(request.form["insulin"]),
            float(request.form["bmi"]),

            # IMPORTANT:
            # Your diabetes HTML should use:
            # name="diabetes_pedigree"

            float(request.form["diabetes_pedigree"]),

            float(request.form["age"])
        ]

        # -------------------------------------------
        # Convert to NumPy array
        # -------------------------------------------

        data = np.array(features).reshape(1, -1)

        # -------------------------------------------
        # Scale input
        # -------------------------------------------

        data = diabetes_scaler.transform(data)

        # -------------------------------------------
        # Prediction
        # -------------------------------------------

        prediction = diabetes_model.predict(data)[0]

        confidence = (
            max(diabetes_model.predict_proba(data)[0]) * 100
        )

        # -------------------------------------------
        # Result
        # -------------------------------------------

        if prediction == 1:

            result = "High Risk of Diabetes"

            risk_color = "red"

            advice = [
                "Consult a doctor for proper diagnosis.",
                "Reduce sugar intake.",
                "Exercise at least 30 minutes daily.",
                "Monitor blood glucose regularly."
            ]

        else:

            result = "Low Risk of Diabetes"

            risk_color = "green"

            advice = [
                "Maintain a healthy lifestyle.",
                "Continue regular exercise.",
                "Eat a balanced diet.",
                "Go for routine health checkups."
            ]

        # -------------------------------------------
        # SAVE REPORT
        # -------------------------------------------

        save_prediction(
            "Diabetes",
            patient_name,
            result,
            confidence,
            advice
        )

        # -------------------------------------------
        # Display result
        # -------------------------------------------

        return render_template(
            "diabetes.html",
            prediction=result,
            confidence=f"{confidence:.2f}",
            risk_color=risk_color,
            advice=advice
        )

    except Exception as e:

        print("DIABETES ERROR:", e)

        return render_template(
            "diabetes.html",
            error=f"Invalid input: {e}"
        )


# ======================================================
# HEART DISEASE MODULE
# ======================================================

@app.route("/heart")
def heart():

    return render_template("heart.html")


@app.route("/predict_heart", methods=["POST"])
def predict_heart():

    try:

        # -------------------------------------------
        # Patient name
        # -------------------------------------------

        patient_name = request.form.get(
            "patient_name",
            "Patient"
        )

        # -------------------------------------------
        # Heart disease features
        # -------------------------------------------

        features = [

            float(request.form["age"]),

            int(request.form["sex"]),

            int(request.form["cp"]),

            float(request.form["trestbps"]),

            float(request.form["chol"]),

            int(request.form["fbs"]),

            int(request.form["restecg"]),

            float(request.form["thalach"]),

            int(request.form["exang"]),

            float(request.form["oldpeak"]),

            int(request.form["slope"]),

            float(request.form["ca"]),

            int(request.form["thal"])
        ]

        # -------------------------------------------
        # Convert to NumPy array
        # -------------------------------------------

        data = np.array(features).reshape(1, -1)

        # -------------------------------------------
        # Scale input
        # -------------------------------------------

        data = heart_scaler.transform(data)

        # -------------------------------------------
        # Prediction
        # -------------------------------------------

        prediction = heart_model.predict(data)[0]

        confidence = (
            max(heart_model.predict_proba(data)[0]) * 100
        )

        # -------------------------------------------
        # Result
        # -------------------------------------------

        if prediction == 1:

            result = "High Risk of Heart Disease"

            risk_color = "red"

            advice = [
                "Consult a cardiologist.",
                "Reduce cholesterol intake.",
                "Exercise regularly.",
                "Monitor blood pressure."
            ]

        else:

            result = "Low Risk of Heart Disease"

            risk_color = "green"

            advice = [
                "Maintain heart-healthy habits.",
                "Eat a balanced diet.",
                "Continue regular exercise.",
                "Attend routine medical checkups."
            ]

        # -------------------------------------------
        # SAVE REPORT
        # -------------------------------------------

        save_prediction(
            "Heart Disease",
            patient_name,
            result,
            confidence,
            advice
        )

        # -------------------------------------------
        # Display result
        # -------------------------------------------

        return render_template(
            "heart.html",
            prediction=result,
            confidence=f"{confidence:.2f}",
            risk_color=risk_color,
            advice=advice
        )

    except Exception as e:

        print("HEART ERROR:", e)

        return render_template(
            "heart.html",
            error=f"Invalid input: {e}"
        )


# ======================================================
# CANCER SCREENING MODULE
# ======================================================

@app.route("/cancer")
def cancer():

    return render_template("cancer.html")


@app.route("/predict_cancer", methods=["POST"])
def predict_cancer():

    try:

        # -------------------------------------------
        # Patient name
        # -------------------------------------------

        patient_name = request.form.get(
            "patient_name",
            "Patient"
        )

        # -------------------------------------------
        # Cancer features
        # -------------------------------------------

        features = [

            float(request.form["radius_mean"]),
            float(request.form["texture_mean"]),
            float(request.form["perimeter_mean"]),
            float(request.form["area_mean"]),
            float(request.form["smoothness_mean"]),
            float(request.form["compactness_mean"]),
            float(request.form["concavity_mean"]),
            float(request.form["concave_points_mean"]),
            float(request.form["symmetry_mean"]),
            float(request.form["fractal_dimension_mean"]),

            float(request.form["radius_se"]),
            float(request.form["texture_se"]),
            float(request.form["perimeter_se"]),
            float(request.form["area_se"]),
            float(request.form["smoothness_se"]),
            float(request.form["compactness_se"]),
            float(request.form["concavity_se"]),
            float(request.form["concave_points_se"]),
            float(request.form["symmetry_se"]),
            float(request.form["fractal_dimension_se"]),

            float(request.form["radius_worst"]),
            float(request.form["texture_worst"]),
            float(request.form["perimeter_worst"]),
            float(request.form["area_worst"]),
            float(request.form["smoothness_worst"]),
            float(request.form["compactness_worst"]),
            float(request.form["concavity_worst"]),
            float(request.form["concave_points_worst"]),
            float(request.form["symmetry_worst"]),
            float(request.form["fractal_dimension_worst"])
        ]

        # -------------------------------------------
        # Convert to NumPy array
        # -------------------------------------------

        data = np.array(features).reshape(1, -1)

        # -------------------------------------------
        # Scale input
        # -------------------------------------------

        data = cancer_scaler.transform(data)

        # -------------------------------------------
        # Prediction
        # -------------------------------------------

        prediction = cancer_model.predict(data)[0]

        # -------------------------------------------
        # Confidence
        # -------------------------------------------

        if hasattr(cancer_model, "predict_proba"):

            confidence = (
                max(cancer_model.predict_proba(data)[0]) * 100
            )

        else:

            confidence = 95.0

        # -------------------------------------------
        # Result
        # -------------------------------------------

        if prediction == 0:

            result = "High Risk (Malignant)"

            risk_color = "red"

            advice = [
                "Consult an oncologist immediately.",
                "Schedule further diagnostic tests.",
                "Do not delay medical evaluation."
            ]

        else:

            result = "Low Risk (Benign)"

            risk_color = "green"

            advice = [
                "Continue regular health checkups.",
                "Maintain a healthy lifestyle.",
                "Follow your doctor's advice."
            ]

        # -------------------------------------------
        # SAVE REPORT
        # -------------------------------------------

        save_prediction(
            "Cancer Screening",
            patient_name,
            result,
            confidence,
            advice
        )

        # -------------------------------------------
        # Display result
        # -------------------------------------------

        return render_template(
            "cancer.html",
            prediction=result,
            confidence=f"{confidence:.2f}",
            risk_color=risk_color,
            advice=advice
        )

    except Exception as e:

        print("CANCER ERROR:", e)

        return render_template(
            "cancer.html",
            error=f"Invalid input: {e}"
        )


# ======================================================
# RUN APPLICATION
# ======================================================

if __name__ == "__main__":

    app.run(debug=True)