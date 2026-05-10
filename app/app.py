"""
Flask version — Fixed:
  1. Images stored in server temp files (not session) → fixes ERR_RESPONSE_HEADERS_TOO_BIG
  2. Upload page is separate from the Name question
  3. Report download added at result page

Run:  python app.py
Then open http://127.0.0.1:5000
"""

import os, pickle, io, base64, uuid, tempfile, json
import numpy as np
import tensorflow as tf
from flask import (Flask, render_template, request, session,
                   redirect, url_for, send_file, abort, make_response)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = BASE
MODELS_DIR = os.path.join(ROOT, "saved_models")

# Temp directory for uploaded images (avoids bloating the session cookie)
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "dermai_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Load models once at startup ────────────────────────────────────────────────
def _load(name):
    with open(os.path.join(MODELS_DIR, name), "rb") as f:
        return pickle.load(f)

sc           = _load("sc.pkl")
orderLabels  = _load("orderLabels.pkl")
ensemble_clf = _load("ensemble_clf.pki")
xgb_clf      = _load("xgb_clf.pki")
lgb_clf      = _load("lgb_clf.pki")
cnn_model    = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, "cnn_tf_model.keras"))

# ── Question definitions (name removed — handled on upload page) ───────────────
QUESTIONS = [
    {"key": "age",                 "type": "slider", "label": "What is the age of the patient?",                              "min": 1,   "max": 100},
    {"key": "gender",              "type": "radio",  "label": "What is the gender of the patient?",                           "options": ["Male", "Female"]},
    {"key": "race",                "type": "radio",  "label": "What is the race of the patient?",                             "options": ["White", "Black", "Hispanic", "Asian or Pacific Islander", "Native American", "Other"]},
    {"key": "region",              "type": "radio",  "label": "What part of the body is the lesion on?",                      "options": ["Abdomen","Arm","Back","Chest","Ear","Face","Foot","Forearm","Hand","Lip","Neck","Nose","Scalp","Thigh"]},
    {"key": "diameter_1",          "type": "number", "label": "What is the largest diameter of the lesion (mm)?",             "min": 0, "max": 9999, "step": "any"},
    {"key": "diameter_2",          "type": "number", "label": "What is the smallest diameter of the lesion (mm)?",            "min": 0, "max": 9999, "step": "any"},
    {"key": "smoke",               "type": "radio",  "label": "Does the patient smoke?",                                      "options": ["True", "False"]},
    {"key": "drink",               "type": "radio",  "label": "Does the patient drink alcohol?",                              "options": ["True", "False"]},
    {"key": "skin_cancer_history", "type": "radio",  "label": "Does the patient have a history of skin cancer?",              "options": ["True", "False"]},
    {"key": "cancer_history",      "type": "radio",  "label": "Does the patient have a history of any other type of cancer?", "options": ["True", "False"]},
    {"key": "pesticide",           "type": "radio",  "label": "Has the patient been exposed to pesticides?",                  "options": ["True", "False"]},
    {"key": "biopsed",             "type": "radio",  "label": "Has the patient's lesion been biopsied?",                      "options": ["True", "False"]},
    {"key": "grew",                "type": "radio",  "label": "Has the lesion grown?",                                        "options": ["True", "False"]},
    {"key": "hurt",                "type": "radio",  "label": "Does the lesion hurt?",                                        "options": ["True", "False"]},
    {"key": "bleed",               "type": "radio",  "label": "Does the lesion bleed?",                                       "options": ["True", "False"]},
    {"key": "itch",                "type": "radio",  "label": "Does the lesion itch?",                                        "options": ["True", "False"]},
    {"key": "elevation",           "type": "radio",  "label": "Is the lesion elevated?",                                      "options": ["True", "False"]},
    {"key": "changed",             "type": "radio",  "label": "Has the lesion changed in appearance?",                        "options": ["True", "False"]},
    {"key": "has_sewage_system",   "type": "radio",  "label": "Does the patient's home have a sewage system?",                "options": ["True", "False"]},
    {"key": "has_piped_water",     "type": "radio",  "label": "Does the patient's home have piped water?",                    "options": ["True", "False"]},
]
TOTAL = len(QUESTIONS)

# ── Image helpers (disk-based, not session) ────────────────────────────────────
def save_image_to_disk(file_storage):
    """Save an uploaded FileStorage to a temp file; return (token, mime)."""
    token    = str(uuid.uuid4())
    mime     = file_storage.mimetype or "image/jpeg"
    ext      = mime.split("/")[-1] if "/" in mime else "jpg"
    path     = os.path.join(UPLOAD_DIR, f"{token}.{ext}")
    file_storage.seek(0)
    file_storage.save(path)
    return token, mime

def get_image_bytes(token, mime):
    """Read image bytes from disk given a token and mime type."""
    ext  = mime.split("/")[-1] if "/" in mime else "jpg"
    path = os.path.join(UPLOAD_DIR, f"{token}.{ext}")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()

def get_image_data_uri(token, mime):
    """Return a data: URI for inline <img> display."""
    data = get_image_bytes(token, mime)
    if data is None:
        return None
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"

# ── Inference ──────────────────────────────────────────────────────────────────
def to_bool(v):
    return 1 if str(v) == "True" else 0

def run_inference(answers, image_bytes):
    img = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (224, 224)) / 255.0
    img = tf.expand_dims(img, axis=0)
    cnn_model.predict(img, verbose=0)

    age_val = float(answers.get("age", 1))
    d1_val  = float(answers.get("diameter_1", 0.0))
    d2_val  = float(answers.get("diameter_2", 0.0))
    scaled  = sc.transform([[age_val, d1_val, d2_val]])
    age_s   = scaled[0][0]

    meta = np.array([[
        age_s,
        1 if answers.get("gender") == "Male" else 0,
        to_bool(answers.get("smoke")),
        to_bool(answers.get("drink")),
        to_bool(answers.get("skin_cancer_history")),
        to_bool(answers.get("cancer_history")),
        to_bool(answers.get("pesticide")),
        to_bool(answers.get("biopsed")),
        to_bool(answers.get("grew")),
        to_bool(answers.get("hurt")),
        to_bool(answers.get("bleed")),
        to_bool(answers.get("itch")),
    ]])

    ensemble_out = ensemble_clf.predict(meta)
    disease      = orderLabels[ensemble_out][0]

    predc = np.array([xgb_clf.predict(meta), lgb_clf.predict(meta)])
    unique_cls, counts = np.unique(predc, return_counts=True)
    freq       = dict(zip(unique_cls, counts))
    pred_class = max(freq, key=freq.get)
    confidence = round(freq[pred_class] / predc.size * 100)

    return disease, confidence

# ── PDF Report ─────────────────────────────────────────────────────────────────
def build_pdf(answers, disease, confidence):
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm,  bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    title_style = ParagraphStyle("title", parent=styles["Title"],
                                  fontSize=20, spaceAfter=6, alignment=TA_CENTER)
    sub_style   = ParagraphStyle("sub",   parent=styles["Normal"],
                                  fontSize=10, textColor=colors.grey,
                                  alignment=TA_CENTER, spaceAfter=18)
    h2_style    = ParagraphStyle("h2",    parent=styles["Heading2"],
                                  fontSize=13, spaceBefore=14, spaceAfter=6,
                                  textColor=colors.HexColor("#4f46e5"))

    story.append(Paragraph("🩺 Skin Disease Prediction Report", title_style))
    story.append(Paragraph("AI-Assisted Dermatology · For clinical review only", sub_style))

    # ── Lesion image ──
    img_token = answers.get("image_token")
    img_mime  = answers.get("image_mime", "image/jpeg")
    if img_token:
        img_bytes = get_image_bytes(img_token, img_mime)
        if img_bytes:
            rl_img = RLImage(io.BytesIO(img_bytes), width=8*cm, height=8*cm)
            rl_img.hAlign = "CENTER"
            story.append(rl_img)
            story.append(Spacer(1, 10))

    # ── Diagnosis block ──
    story.append(Paragraph("Diagnosis", h2_style))
    diag_data = [
        ["Predicted Disease", disease],
        ["Confidence",        f"{confidence}%"],
    ]
    t = Table(diag_data, colWidths=[6*cm, 10*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#ede9fe")),
        ("BACKGROUND",  (0, 1), (-1, 1), colors.HexColor("#f5f3ff")),
        ("TEXTCOLOR",   (0, 0), (0, -1), colors.HexColor("#4f46e5")),
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 11),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#ede9fe"), colors.HexColor("#f5f3ff")]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#c7d2fe")),
        ("PADDING",     (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # ── Patient info ──
    story.append(Paragraph("Patient Information", h2_style))

    def yn(v):
        return "Yes" if str(v) == "True" else ("No" if str(v) == "False" else str(v))

    rows = [
        ["Field", "Value"],
        ["Name",                  answers.get("name", "N/A")],
        ["Age",                   str(int(float(answers.get("age", 0)))) + " yrs"],
        ["Gender",                answers.get("gender", "N/A")],
        ["Race",                  answers.get("race", "N/A")],
        ["Lesion Region",         answers.get("region", "N/A")],
        ["Largest Diameter",      f"{answers.get('diameter_1', 'N/A')} mm"],
        ["Smallest Diameter",     f"{answers.get('diameter_2', 'N/A')} mm"],
        ["Smokes",                yn(answers.get("smoke"))],
        ["Drinks Alcohol",        yn(answers.get("drink"))],
        ["Skin Cancer History",   yn(answers.get("skin_cancer_history"))],
        ["Cancer History",        yn(answers.get("cancer_history"))],
        ["Pesticide Exposure",    yn(answers.get("pesticide"))],
        ["Biopsied",              yn(answers.get("biopsed"))],
        ["Lesion Grew",           yn(answers.get("grew"))],
        ["Lesion Hurts",          yn(answers.get("hurt"))],
        ["Lesion Bleeds",         yn(answers.get("bleed"))],
        ["Lesion Itches",         yn(answers.get("itch"))],
        ["Elevated",              yn(answers.get("elevation"))],
        ["Changed",               yn(answers.get("changed"))],
        ["Has Sewage System",     yn(answers.get("has_sewage_system"))],
        ["Has Piped Water",       yn(answers.get("has_piped_water"))],
    ]

    pt = Table(rows, colWidths=[7*cm, 9*cm])
    row_colors = [colors.HexColor("#f5f5ff") if i % 2 == 0 else colors.white
                  for i in range(len(rows))]
    pt.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f5f5ff"), colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#e0e0f0")),
        ("PADDING",     (0, 0), (-1, -1), 7),
    ]))
    story.append(pt)
    story.append(Spacer(1, 16))

    disclaimer = ("⚠️  This report is generated by an AI model and is intended for "
                  "informational purposes only. It must be reviewed by a qualified "
                  "dermatologist before any clinical decision is made.")
    story.append(Paragraph(disclaimer,
                            ParagraphStyle("disc", parent=styles["Normal"],
                                           fontSize=8, textColor=colors.grey,
                                           borderPad=6)))

    doc.build(story)
    buf.seek(0)
    return buf

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=os.path.join(BASE, "templates"))
app.secret_key = "dermai-secret-2024"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# ── Upload page (new first step) ───────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    session.clear()
    session["answers"] = {}
    return redirect(url_for("upload"))

@app.route("/upload", methods=["GET", "POST"])
def upload():
    error = None
    answers = session.get("answers", {})

    if request.method == "POST":
        # Patient name
        name = request.form.get("name", "").strip()
        if not name:
            error = "Please enter the patient's name."
        else:
            answers["name"] = name

        # Lesion image
        if not error:
            if "image" in request.files and request.files["image"].filename:
                token, mime = save_image_to_disk(request.files["image"])
                # Delete old temp file if any
                old_token = answers.get("image_token")
                old_mime  = answers.get("image_mime", "image/jpeg")
                if old_token:
                    old_ext  = old_mime.split("/")[-1] if "/" in old_mime else "jpg"
                    old_path = os.path.join(UPLOAD_DIR, f"{old_token}.{old_ext}")
                    if os.path.exists(old_path):
                        os.remove(old_path)
                answers["image_token"] = token
                answers["image_mime"]  = mime
            elif "image_token" not in answers:
                error = "Please upload a skin lesion image."

        # Optional patient photo
        if not error:
            if "patient_photo" in request.files and request.files["patient_photo"].filename:
                p_token, p_mime = save_image_to_disk(request.files["patient_photo"])
                answers["photo_token"] = p_token
                answers["photo_mime"]  = p_mime

        if not error:
            session["answers"] = answers
            session["step"]    = 0
            return redirect(url_for("question"))

    # Build preview data URIs for template (small — only for display, not stored in session)
    image_preview = None
    photo_preview = None
    if "image_token" in answers:
        image_preview = get_image_data_uri(answers["image_token"], answers.get("image_mime","image/jpeg"))
    if "photo_token" in answers:
        photo_preview = get_image_data_uri(answers["photo_token"], answers.get("photo_mime","image/jpeg"))

    return render_template("index.html",
                           page          = "upload",
                           answers       = answers,
                           error         = error,
                           image_preview = image_preview,
                           photo_preview = photo_preview)

# ── Questionnaire ──────────────────────────────────────────────────────────────
@app.route("/question", methods=["GET", "POST"])
def question():
    if "answers" not in session:
        return redirect(url_for("index"))

    step    = session.get("step", 0)
    answers = session.get("answers", {})
    error   = None

    if request.method == "POST":
        action = request.form.get("action", "next")
        q      = QUESTIONS[step]
        key    = q["key"]

        if action == "prev":
            # If on first question, go back to upload page
            if step == 0:
                return redirect(url_for("upload"))
            session["step"] = step - 1
            return redirect(url_for("question"))

        # Validate & save
        if q["type"] == "radio":
            val = request.form.get(key)
            if not val:
                error = "Please select an option before continuing."
            else:
                answers[key] = val

        elif q["type"] == "text":
            val = request.form.get(key, "").strip()
            if not val:
                error = "Please enter a value before continuing."
            else:
                answers[key] = val

        elif q["type"] in ("number", "slider"):
            val = request.form.get(key)
            if val is None or val == "":
                error = "Please enter a value before continuing."
            else:
                try:
                    answers[key] = float(val)
                except ValueError:
                    error = "Please enter a valid number."

        if not error:
            session["answers"] = answers
            if action == "next":
                if step < TOTAL - 1:
                    session["step"] = step + 1
                    return redirect(url_for("question"))
                else:
                    return redirect(url_for("result"))

    step = session.get("step", 0)
    q    = QUESTIONS[step]
    pct  = round((step + 1) / TOTAL * 100)

    # Build small lesion preview for the sidebar
    answers = session.get("answers", {})
    image_preview = None
    if "image_token" in answers:
        image_preview = get_image_data_uri(answers["image_token"],
                                           answers.get("image_mime","image/jpeg"))

    return render_template("index.html",
                           page          = "question",
                           q             = q,
                           step          = step,
                           total         = TOTAL,
                           pct           = pct,
                           answers       = answers,
                           error         = error,
                           image_preview = image_preview)

# ── Result ─────────────────────────────────────────────────────────────────────
@app.route("/result")
def result():
    answers = session.get("answers", {})
    if "image_token" not in answers:
        return redirect(url_for("index"))

    image_bytes = get_image_bytes(answers["image_token"], answers.get("image_mime","image/jpeg"))
    if image_bytes is None:
        return redirect(url_for("index"))

    disease, confidence = run_inference(answers, image_bytes)
    # Cache result in session for the PDF download route
    session["result_disease"]    = disease
    session["result_confidence"] = confidence

    r    = 54
    circ = 2 * 3.14159 * r
    dash = circ * confidence / 100
    gap  = circ - dash

    image_src = get_image_data_uri(answers["image_token"], answers.get("image_mime","image/jpeg"))
    photo_src = None
    if "photo_token" in answers:
        photo_src = get_image_data_uri(answers["photo_token"], answers.get("photo_mime","image/jpeg"))

    return render_template("index.html",
                           page       = "result",
                           answers    = answers,
                           disease    = disease,
                           confidence = confidence,
                           dash       = round(dash, 1),
                           gap        = round(gap,  1),
                           r          = r,
                           image_src  = image_src,
                           photo_src  = photo_src)

# ── PDF download ───────────────────────────────────────────────────────────────
@app.route("/download-report")
def download_report():
    answers    = session.get("answers", {})
    disease    = session.get("result_disease")
    confidence = session.get("result_confidence")

    if not answers or not disease:
        return redirect(url_for("index"))

    pdf_buf  = build_pdf(answers, disease, confidence)
    pdf_data = pdf_buf.read()
    patient  = answers.get("name", "patient").replace(" ", "_")
    filename = f"DermAI_Report_{patient}.pdf"

    response = make_response(pdf_data)
    response.headers["Content-Type"]        = "application/pdf"
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    response.headers["Content-Length"]      = len(pdf_data)
    response.headers["Cache-Control"]       = "no-cache, no-store, must-revalidate"
    return response

# ── Reset ──────────────────────────────────────────────────────────────────────
@app.route("/reset")
def reset():
    # Clean up temp images
    answers = session.get("answers", {})
    for key in ("image_token", "photo_token"):
        token = answers.get(key)
        mime  = answers.get(key.replace("token","mime"), "image/jpeg")
        if token:
            ext  = mime.split("/")[-1] if "/" in mime else "jpg"
            path = os.path.join(UPLOAD_DIR, f"{token}.{ext}")
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
    session.clear()
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)