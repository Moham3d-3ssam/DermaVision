"""
Flask version — Fixed:
  1. Images stored in server temp files (not session) → fixes ERR_RESPONSE_HEADERS_TOO_BIG
  2. Upload page is separate from the Name question
  3. Report download added at result page
  4. Prediction pipeline now matches notebook 06 exactly:
         final_x = [ CNN softmax (6) | ANN softmax (6) * 2 ]  ->  ensemble_clf
     (before, the 12 patient-metadata values were fed straight into the
      ensemble, which was trained on those 12 CNN+ANN probabilities).
  5. Confidence = real ensemble probability of the predicted class.
  6. Prediction cached (no re-run on refresh), validated uploads, auto-cleanup.

Run:  python app.py
Then open http://127.0.0.1:5000

Production:  gunicorn -w 1 --threads 4 --timeout 120 -b 0.0.0.0:$PORT app:app
"""

import io
import math
import os
import pickle
import re
import secrets
import tempfile
import threading
import time
import uuid

import numpy as np
from flask import (Flask, abort, make_response, redirect, render_template,
                   request, send_file, session, url_for)
from PIL import Image, ImageOps

try:                      # same library the notebook used for preprocessing
    import cv2
except ImportError:       # falls back to Pillow (tiny numeric differences)
    cv2 = None

Image.MAX_IMAGE_PIXELS = 50_000_000          # decompression-bomb guard

# ── Paths / config ─────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = next(
    (p for p in (os.environ.get("MODELS_DIR", ""),
                 os.path.join(BASE, "saved_models"),                    # next to app.py
                 os.path.join(os.path.dirname(BASE), "saved_models"))   # repo root (DermaVision/saved_models)
     if p and os.path.isdir(p)),
    os.path.join(BASE, "saved_models"))
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "dermai_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_BYTES = 16 * 1024 * 1024            # per image
MAX_SIDE       = 1024                        # stored image size
FILE_TTL       = 60 * 60                     # delete temp images after 1 hour
TOKEN_RE       = re.compile(r"[0-9a-f]{32}")


# ── Models (loaded once at startup) ────────────────────────────────────────────
def _load(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


sc           = _load("sc.pkl")
orderLabels  = _load("orderLabels.pkl")
ensemble_clf = _load("ensemble_clf.pki")      # VotingClassifier(xgb, lgb, gb) — soft voting

# Order of the ANN input vector = ann_final.drop(columns=["diagnostic"]).columns  (verified)
REGION_COLS = ["ARM", "BACK", "CHEST", "EAR", "FACE", "FOOT", "FOREARM",
               "HAND", "LIP", "NECK", "NOSE", "SCALP", "THIGH"]       # ABDOMEN = all zeros (drop_first)
ANN_FEATURES = (["age", "gender", "smoke", "drink", "skin_cancer_history", "cancer_history",
                 "pesticide", "biopsed", "grew", "hurt", "bleed", "itch", "elevation", "changed",
                 "diameter_1", "diameter_2", "has_sewage_system", "has_piped_water", "race"]
                + [f"region_{r}" for r in REGION_COLS])               # 32 columns

# The training data has a numeric `race` column with values 1..5. Its meaning is NOT documented
# in the notebooks, so this mapping is a best guess — edit it if you know the real coding.
RACE_CODE = {"White": 1, "Black": 2, "Hispanic": 3, "Asian or Pacific Islander": 4,
             "Native American": 5, "Other": 3}

_cnn_model = _ann_model = None
_load_lock    = threading.Lock()
_predict_lock = threading.Lock()


def get_models():
    """Load the two Keras models once (thread-safe) and sanity-check their shapes."""
    global _cnn_model, _ann_model
    if _cnn_model is None or _ann_model is None:
        with _load_lock:
            if _cnn_model is None or _ann_model is None:
                import tensorflow as tf
                cnn = tf.keras.models.load_model(os.path.join(MODELS_DIR, "cnn_tf_model.keras"))
                ann = tf.keras.models.load_model(os.path.join(MODELS_DIR, "ann_model.keras"))

                n_ann = ann.input_shape[-1]
                if n_ann != len(ANN_FEATURES):
                    raise RuntimeError(
                        f"ann_model expects {n_ann} input features but ANN_FEATURES has "
                        f"{len(ANN_FEATURES)}. Fix ANN_FEATURES / build_ann_vector() to match "
                        f"ann_final.columns from the notebook.")
                n_ens = getattr(ensemble_clf, "n_features_in_", None)
                n_out = cnn.output_shape[-1] + ann.output_shape[-1]
                if n_ens is not None and n_ens != n_out:
                    raise RuntimeError(
                        f"ensemble_clf expects {n_ens} features but CNN+ANN output {n_out}.")
                _cnn_model, _ann_model = cnn, ann
    return _cnn_model, _ann_model


if os.environ.get("PRELOAD_MODELS", "1") == "1":
    get_models()


# ── Question definitions (name removed — handled on upload page) ───────────────
YES_NO = ["True", "False"]
QUESTIONS = [
    {"key": "age",                 "type": "slider", "label": "What is the age of the patient?",                              "min": 1, "max": 100},
    {"key": "gender",              "type": "radio",  "label": "What is the gender of the patient?",                           "options": ["Male", "Female"]},
    {"key": "race",                "type": "radio",  "label": "What is the race of the patient?",                             "options": ["White", "Black", "Hispanic", "Asian or Pacific Islander", "Native American", "Other"]},
    {"key": "region",              "type": "radio",  "label": "What part of the body is the lesion on?",                      "options": ["Abdomen", "Arm", "Back", "Chest", "Ear", "Face", "Foot", "Forearm", "Hand", "Lip", "Neck", "Nose", "Scalp", "Thigh"]},
    {"key": "diameter_1",          "type": "number", "label": "What is the largest diameter of the lesion (mm)?",             "min": 0, "max": 9999, "step": "any"},
    {"key": "diameter_2",          "type": "number", "label": "What is the smallest diameter of the lesion (mm)?",            "min": 0, "max": 9999, "step": "any"},
    {"key": "smoke",               "type": "radio",  "label": "Does the patient smoke?",                                      "options": YES_NO},
    {"key": "drink",               "type": "radio",  "label": "Does the patient drink alcohol?",                              "options": YES_NO},
    {"key": "skin_cancer_history", "type": "radio",  "label": "Does the patient have a history of skin cancer?",              "options": YES_NO},
    {"key": "cancer_history",      "type": "radio",  "label": "Does the patient have a history of any other type of cancer?", "options": YES_NO},
    {"key": "pesticide",           "type": "radio",  "label": "Has the patient been exposed to pesticides?",                  "options": YES_NO},
    {"key": "biopsed",             "type": "radio",  "label": "Has the patient's lesion been biopsied?",                      "options": YES_NO},
    {"key": "grew",                "type": "radio",  "label": "Has the lesion grown?",                                        "options": YES_NO},
    {"key": "hurt",                "type": "radio",  "label": "Does the lesion hurt?",                                        "options": YES_NO},
    {"key": "bleed",               "type": "radio",  "label": "Does the lesion bleed?",                                       "options": YES_NO},
    {"key": "itch",                "type": "radio",  "label": "Does the lesion itch?",                                        "options": YES_NO},
    {"key": "elevation",           "type": "radio",  "label": "Is the lesion elevated?",                                      "options": YES_NO},
    {"key": "changed",             "type": "radio",  "label": "Has the lesion changed in appearance?",                        "options": YES_NO},
    {"key": "has_sewage_system",   "type": "radio",  "label": "Does the patient's home have a sewage system?",                "options": YES_NO},
    {"key": "has_piped_water",     "type": "radio",  "label": "Does the patient's home have piped water?",                    "options": YES_NO},
]
TOTAL = len(QUESTIONS)


# ── Image helpers (disk-based, never in the session cookie) ────────────────────
def image_path(token):
    if not token or not TOKEN_RE.fullmatch(token):
        return None
    return os.path.join(UPLOAD_DIR, f"{token}.jpg")


def array_path(token):
    p = image_path(token)
    return p[:-4] + ".npy" if p else None


def image_exists(token):
    p = image_path(token)
    return bool(p and os.path.exists(p))


def delete_image(token):
    for p in (image_path(token), array_path(token)):
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def cnn_array_from_bytes(data, pil_img):
    """224x224 RGB uint8 — same as the notebook: cv2.imread -> cv2.resize -> BGR2RGB."""
    if cv2 is not None:
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    return np.asarray(pil_img.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.uint8)


def save_image(file_storage, for_cnn=False):
    """Validate + normalise an upload. Returns a token, or raises ValueError."""
    data = file_storage.read()
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("Image is too large (max 16 MB).")
    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in ("PNG", "JPEG"):
            raise ValueError("Only PNG and JPG images are allowed.")
        img.load()
        cnn_arr = cnn_array_from_bytes(data, img) if for_cnn else None
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    except ValueError:
        raise
    except Exception:
        raise ValueError("That file is not a valid image.")

    token = uuid.uuid4().hex
    img.save(image_path(token), "JPEG", quality=90, optimize=True)
    if cnn_arr is not None:
        np.save(array_path(token), cnn_arr)
    return token


_last_cleanup = 0.0


def cleanup_old_files():
    """Delete temp images older than FILE_TTL (throttled to once / 5 min)."""
    global _last_cleanup
    now = time.time()
    if now - _last_cleanup < 300:
        return
    _last_cleanup = now
    try:
        for entry in os.scandir(UPLOAD_DIR):
            if entry.is_file() and now - entry.stat().st_mtime > FILE_TTL:
                try:
                    os.remove(entry.path)
                except OSError:
                    pass
    except OSError:
        pass


def delete_session_images(answers):
    for key in ("image_token", "photo_token"):
        delete_image(answers.get(key))


def image_url(kind, answers):
    key = "image_token" if kind == "lesion" else "photo_token"
    token = answers.get(key)
    if token and image_exists(token):
        return url_for("serve_image", kind=kind, v=token)
    return None


# ── Inference ──────────────────────────────────────────────────────────────────
DISEASE_NAMES = {
    "ACK": "Actinic Keratosis",
    "BCC": "Basal Cell Carcinoma",
    "MEL": "Melanoma",
    "NEV": "Nevus",
    "SCC": "Squamous Cell Carcinoma",
    "SEK": "Seborrheic Keratosis",
}


def to_bool(v):
    return 1 if str(v) == "True" else 0


def build_ann_vector(answers):
    """Patient metadata -> the 32-column ANN input (same order/encoding as notebooks 02/05)."""
    scaled = sc.transform([[float(answers.get("age", 1)),
                            float(answers.get("diameter_1", 0.0)),
                            float(answers.get("diameter_2", 0.0))]])[0]          # age, d1, d2 (z-scored)
    region = str(answers.get("region", "")).upper()
    row = {
        "age": scaled[0],
        "gender": 1 if answers.get("gender") == "Male" else 0,                  # MALE:1 FEMALE:0
        "diameter_1": scaled[1],
        "diameter_2": scaled[2],
        "race": RACE_CODE.get(answers.get("race"), 3),
    }
    for k in ("smoke", "drink", "skin_cancer_history", "cancer_history", "pesticide", "biopsed",
              "grew", "hurt", "bleed", "itch", "elevation", "changed",
              "has_sewage_system", "has_piped_water"):
        row[k] = to_bool(answers.get(k))                                        # TRUE:1 FALSE:0
    for r in REGION_COLS:
        row[f"region_{r}"] = 1 if region == r else 0
    return np.array([[row[c] for c in ANN_FEATURES]], dtype=np.float32)


def _np(out):
    return out.numpy() if hasattr(out, "numpy") else np.asarray(out)


def load_cnn_input(answers):
    path = array_path(answers["image_token"])
    if path and os.path.exists(path):
        arr = np.load(path)
    else:                                   # fallback: rebuild from the stored JPEG
        with Image.open(image_path(answers["image_token"])) as im:
            arr = np.asarray(im.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.uint8)
    return (arr.astype("float32") / 255.0)[None, ...]


def run_inference(answers):
    cnn, ann = get_models()
    with _predict_lock:
        cnn_p = _np(cnn(load_cnn_input(answers), training=False))                    # (1, 6)
        ann_p = _np(ann(build_ann_vector(answers), training=False)) * 2              # (1, 6)  ← same *2 as notebook 06

    final_x = np.concatenate([cnn_p, ann_p], axis=1)                                 # (1, 12)

    ensemble_out = ensemble_clf.predict(final_x)
    proba        = ensemble_clf.predict_proba(final_x)[0]
    code         = str(orderLabels[ensemble_out][0])
    confidence   = proba[list(ensemble_clf.classes_).index(ensemble_out[0])] * 100

    name    = DISEASE_NAMES.get(code)
    disease = f"{name} ({code})" if name else code
    return disease, int(round(float(confidence)))


# ── PDF Report (server-side; the result page uses it as a fallback) ───────────
def build_pdf(answers, disease, confidence):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image as RLImage, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=20,
                                 spaceAfter=6, alignment=TA_CENTER)
    sub_style   = ParagraphStyle("sub", parent=styles["Normal"], fontSize=10,
                                 textColor=colors.grey, alignment=TA_CENTER, spaceAfter=18)
    h2_style    = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, spaceBefore=14,
                                 spaceAfter=6, textColor=colors.HexColor("#4f46e5"))

    story.append(Paragraph("Skin Disease Prediction Report", title_style))
    story.append(Paragraph("AI-Assisted Dermatology · For clinical review only", sub_style))

    # lesion image (keep aspect ratio)
    path = image_path(answers.get("image_token"))
    if path and os.path.exists(path):
        with Image.open(path) as im:
            w, h = im.size
        scale = min(8*cm / w, 8*cm / h)
        rl_img = RLImage(path, width=w*scale, height=h*scale)
        rl_img.hAlign = "CENTER"
        story += [rl_img, Spacer(1, 10)]

    story.append(Paragraph("Diagnosis", h2_style))
    t = Table([["Predicted Disease", disease], ["Confidence", f"{confidence}%"]],
              colWidths=[6*cm, 10*cm])
    t.setStyle(TableStyle([
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#4f46e5")),
        ("FONTNAME",  (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, -1), 11),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#ede9fe"), colors.HexColor("#f5f3ff")]),
        ("GRID",      (0, 0), (-1, -1), 0.5, colors.HexColor("#c7d2fe")),
        ("PADDING",   (0, 0), (-1, -1), 8),
    ]))
    story += [t, Spacer(1, 12), Paragraph("Patient Information", h2_style)]

    def yn(v):
        return "Yes" if str(v) == "True" else ("No" if str(v) == "False" else str(v))

    rows = [
        ["Field", "Value"],
        ["Name",                answers.get("name", "N/A")],
        ["Age",                 str(int(float(answers.get("age", 0)))) + " yrs"],
        ["Gender",              answers.get("gender", "N/A")],
        ["Race",                answers.get("race", "N/A")],
        ["Lesion Region",       answers.get("region", "N/A")],
        ["Largest Diameter",    f"{answers.get('diameter_1', 'N/A')} mm"],
        ["Smallest Diameter",   f"{answers.get('diameter_2', 'N/A')} mm"],
        ["Smokes",              yn(answers.get("smoke"))],
        ["Drinks Alcohol",      yn(answers.get("drink"))],
        ["Skin Cancer History", yn(answers.get("skin_cancer_history"))],
        ["Cancer History",      yn(answers.get("cancer_history"))],
        ["Pesticide Exposure",  yn(answers.get("pesticide"))],
        ["Biopsied",            yn(answers.get("biopsed"))],
        ["Lesion Grew",         yn(answers.get("grew"))],
        ["Lesion Hurts",        yn(answers.get("hurt"))],
        ["Lesion Bleeds",       yn(answers.get("bleed"))],
        ["Lesion Itches",       yn(answers.get("itch"))],
        ["Elevated",            yn(answers.get("elevation"))],
        ["Changed",             yn(answers.get("changed"))],
        ["Has Sewage System",   yn(answers.get("has_sewage_system"))],
        ["Has Piped Water",     yn(answers.get("has_piped_water"))],
    ]
    from xml.sax.saxutils import escape      # Paragraph parses <tags>, so escape user text
    pt = Table([[Paragraph(escape(str(c)), styles["Normal"]) if j == 1 else c for j, c in enumerate(r)]
                for r in rows], colWidths=[7*cm, 9*cm])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f5f5ff"), colors.white]),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#e0e0f0")),
        ("PADDING",    (0, 0), (-1, -1), 7),
    ]))
    story += [pt, Spacer(1, 16), Paragraph(
        "This report is generated by an AI model and is intended for informational purposes "
        "only. It must be reviewed by a qualified dermatologist before any clinical decision "
        "is made.",
        ParagraphStyle("disc", parent=styles["Normal"], fontSize=8, textColor=colors.grey))]

    doc.build(story)
    buf.seek(0)
    return buf


# ── Flask app ──────────────────────────────────────────────────────────────────
def _secret_key():
    key = os.environ.get("SECRET_KEY")
    if key:
        return key
    path = os.path.join(tempfile.gettempdir(), "dermai_secret_key")
    try:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(secrets.token_hex(32))
        except FileExistsError:
            pass
        with open(path) as f:
            stored = f.read().strip()
        if stored:
            return stored
    except OSError:
        pass
    return secrets.token_hex(32)


app = Flask(__name__, template_folder=os.path.join(BASE, "templates"))
app.secret_key = _secret_key()
app.config.update(
    MAX_CONTENT_LENGTH=2 * MAX_FILE_BYTES + 1024 * 1024,   # two images + form
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)


def render_upload(answers, error=None, status=200):
    return render_template("index.html",
                           page="upload",
                           answers=answers,
                           error=error,
                           image_preview=image_url("lesion", answers),
                           photo_preview=image_url("photo", answers)), status


@app.errorhandler(413)
def too_large(_e):
    return render_upload(session.get("answers", {}),
                         "Upload is too large (max 16 MB per image).", 413)


# ── Pages ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    delete_session_images(session.get("answers", {}))
    session.clear()
    session["answers"] = {}
    return redirect(url_for("upload"))


@app.route("/upload", methods=["GET", "POST"])
def upload():
    cleanup_old_files()
    answers = session.get("answers", {})
    error = None

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if not name:
            error = "Please enter the patient's name."
        elif len(name) > 100:
            error = "Patient name is too long (max 100 characters)."

        new_image = new_photo = None
        if not error:
            try:
                f = request.files.get("image")
                if f and f.filename:
                    new_image = save_image(f, for_cnn=True)
                elif not image_exists(answers.get("image_token")):
                    error = "Please upload a skin lesion image."

                p = request.files.get("patient_photo")
                if not error and p and p.filename:
                    new_photo = save_image(p)
            except ValueError as e:
                error = str(e)
                delete_image(new_image)

        if not error:
            answers["name"] = name
            if new_image:
                delete_image(answers.get("image_token"))
                answers["image_token"] = new_image
            if new_photo:
                delete_image(answers.get("photo_token"))
                answers["photo_token"] = new_photo
            session["answers"] = answers
            session["step"] = 0
            session.pop("result", None)
            return redirect(url_for("question"))

    return render_upload(answers, error)


@app.route("/question", methods=["GET", "POST"])
def question():
    answers = session.get("answers") or {}
    if not answers.get("name") or not image_exists(answers.get("image_token")):
        return redirect(url_for("upload"))

    try:
        step = int(session.get("step", 0))
    except (TypeError, ValueError):
        step = 0
    step = max(0, min(step, TOTAL - 1))
    error = None

    if request.method == "POST":
        action = request.form.get("action", "next")
        q, key = QUESTIONS[step], QUESTIONS[step]["key"]

        if action == "prev":
            if step == 0:
                return redirect(url_for("upload"))
            session["step"] = step - 1
            return redirect(url_for("question"))

        if q["type"] == "radio":
            val = request.form.get(key)
            if not val:
                error = "Please select an option before continuing."
            elif val not in q["options"]:
                error = "Invalid option selected."
            else:
                answers[key] = val

        elif q["type"] in ("number", "slider"):
            raw = request.form.get(key, "").strip()
            if raw == "":
                error = "Please enter a value before continuing."
            else:
                try:
                    num = float(raw)
                except ValueError:
                    num = None
                if num is None or not math.isfinite(num):
                    error = "Please enter a valid number."
                elif not (q["min"] <= num <= q["max"]):
                    error = f"Value must be between {q['min']} and {q['max']}."
                elif key == "diameter_2" and num > float(answers.get("diameter_1", num)):
                    error = "The smallest diameter can't be larger than the largest diameter."
                else:
                    answers[key] = num

        if not error:
            session["answers"] = answers
            session.pop("result", None)          # answers changed -> recompute
            if step < TOTAL - 1:
                session["step"] = step + 1
                return redirect(url_for("question"))
            return redirect(url_for("result"))

    return render_template("index.html",
                           page="question",
                           q=QUESTIONS[step],
                           step=step,
                           total=TOTAL,
                           pct=round((step + 1) / TOTAL * 100),
                           answers=answers,
                           error=error,
                           image_preview=image_url("lesion", answers))


@app.route("/result")
def result():
    answers = session.get("answers") or {}
    if not image_exists(answers.get("image_token")):
        return redirect(url_for("index"))

    # all questions must be answered
    for i, q in enumerate(QUESTIONS):
        if q["key"] not in answers:
            session["step"] = i
            return redirect(url_for("question"))

    cached = session.get("result")
    if cached:
        disease, confidence = cached["disease"], cached["confidence"]
    else:
        try:
            disease, confidence = run_inference(answers)
        except Exception:
            app.logger.exception("Inference failed")
            return render_upload(answers, "Prediction failed. Please try again.", 500)
        # Cache result in session for the PDF download route (and for page refreshes)
        session["result"] = {"disease": disease, "confidence": confidence}

    flag_keys = ("smoke", "drink", "skin_cancer_history", "cancer_history", "pesticide",
                 "biopsed", "grew", "hurt", "bleed", "itch", "elevation", "changed",
                 "has_sewage_system", "has_piped_water")
    try:
        age_txt = f"{int(float(answers.get('age', 0)))} yrs"
    except (TypeError, ValueError):
        age_txt = "N/A"
    report = {                                  # consumed by the jsPDF code in the page
        "disease": disease,
        "name":    answers.get("name") or "patient",
        "age":     age_txt,
        "gender":  answers.get("gender", "N/A"),
        "race":    answers.get("race", "N/A"),
        "region":  answers.get("region", "N/A"),
        "d1":      f"{answers.get('diameter_1', 'N/A')} mm",
        "d2":      f"{answers.get('diameter_2', 'N/A')} mm",
        "flags":   {k: answers.get(k, "N/A") for k in flag_keys},
    }

    return render_template("index.html",
                           page="result",
                           answers=answers,
                           report=report,
                           disease=disease,
                           confidence=confidence,
                           image_src=image_url("lesion", answers),
                           photo_src=image_url("photo", answers))


@app.route("/image/<kind>")
def serve_image(kind):
    key = {"lesion": "image_token", "photo": "photo_token"}.get(kind)
    if not key:
        abort(404)
    token = (session.get("answers") or {}).get(key)
    if not image_exists(token):
        abort(404)
    resp = send_file(image_path(token), mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "private, max-age=3600"   # URL carries ?v=token
    return resp


# ── PDF download ───────────────────────────────────────────────────────────────
@app.route("/download-report")
def download_report():
    answers = session.get("answers", {})
    cached  = session.get("result")          # cached by /result
    if not answers or not cached or not image_exists(answers.get("image_token")):
        return redirect(url_for("index"))

    pdf_data = build_pdf(answers, cached["disease"], cached["confidence"]).read()
    patient  = re.sub(r"[^\w\-]+", "_", answers.get("name", "patient")).strip("_") or "patient"

    response = make_response(pdf_data)
    response.headers["Content-Type"]        = "application/pdf"
    response.headers["Content-Disposition"] = f'attachment; filename="DermAI_Report_{patient}.pdf"'
    response.headers["Content-Length"]      = len(pdf_data)
    response.headers["Cache-Control"]       = "no-cache, no-store, must-revalidate"
    return response


@app.route("/reset")
def reset():
    delete_session_images(session.get("answers", {}))
    session.clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
