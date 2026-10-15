"""
Generate the technical report PDF using reportlab.
Run: python scripts/generate_report.py
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Frame, PageTemplate
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus.doctemplate import BaseDocTemplate

OUTPUT = Path("docs/technical_report.pdf")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# ── Colours ──────────────────────────────────────────────────────────────────
BLUE   = colors.HexColor("#1a4f8a")
LBLUE  = colors.HexColor("#dce8f5")
LGRAY  = colors.HexColor("#f4f4f4")
DGRAY  = colors.HexColor("#333333")
MGRAY  = colors.HexColor("#666666")
WHITE  = colors.white

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "ReportTitle",
    parent=styles["Title"],
    fontSize=20,
    textColor=BLUE,
    spaceAfter=8,
    alignment=TA_CENTER,
    leading=26,
    fontName="Helvetica-Bold",
)
subtitle_style = ParagraphStyle(
    "Subtitle",
    parent=styles["Normal"],
    fontSize=11,
    textColor=MGRAY,
    alignment=TA_CENTER,
    spaceAfter=4,
)
section_style = ParagraphStyle(
    "SectionHeading",
    parent=styles["Heading1"],
    fontSize=13,
    textColor=BLUE,
    spaceBefore=16,
    spaceAfter=6,
    fontName="Helvetica-Bold",
)
subsection_style = ParagraphStyle(
    "SubHeading",
    parent=styles["Normal"],
    fontSize=11,
    textColor=BLUE,
    spaceBefore=8,
    spaceAfter=4,
    fontName="Helvetica-Bold",
)
body_style = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontSize=10,
    leading=15,
    alignment=TA_JUSTIFY,
    spaceAfter=5,
    textColor=DGRAY,
)
bullet_style = ParagraphStyle(
    "Bullet",
    parent=body_style,
    leftIndent=18,
    bulletIndent=8,
    spaceAfter=3,
    alignment=TA_LEFT,
)
code_style = ParagraphStyle(
    "Code",
    parent=styles["Normal"],
    fontSize=9,
    leading=13,
    fontName="Courier",
    textColor=DGRAY,
    leftIndent=12,
    spaceAfter=4,
    backColor=LGRAY,
)
caption_style = ParagraphStyle(
    "Caption",
    parent=styles["Normal"],
    fontSize=8,
    textColor=MGRAY,
    alignment=TA_CENTER,
    spaceAfter=6,
)
footer_style = ParagraphStyle(
    "Footer",
    parent=styles["Normal"],
    fontSize=8,
    textColor=MGRAY,
    alignment=TA_CENTER,
)

def H(text):  return Paragraph(text, section_style)
def SH(text): return Paragraph(text, subsection_style)
def P(text):  return Paragraph(text, body_style)
def B(text):  return Paragraph(f"&#8226; {text}", bullet_style)
def C(text):  return Paragraph(text, code_style)
def SP(n=6):  return Spacer(1, n)
def HR():     return HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=6)

def make_table(data, col_widths, header_bg=BLUE):
    t = Table(data, colWidths=col_widths)
    n = len(data)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  header_bg),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LGRAY, WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.grey),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    return t

# ── Page callback ─────────────────────────────────────────────────────────────
TITLE_SHORT = "Chest X-Ray Pneumonia Detection — Technical Report"

def on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    # Header (skip title page)
    if doc.page > 1:
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MGRAY)
        canvas.drawString(2.5*cm, h - 1.8*cm, TITLE_SHORT)
        canvas.setStrokeColor(colors.lightgrey)
        canvas.setLineWidth(0.4)
        canvas.line(2.5*cm, h - 2.0*cm, w - 2.5*cm, h - 2.0*cm)
    # Footer
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MGRAY)
    canvas.drawCentredString(w / 2, 1.5*cm, f"Page {doc.page}")
    canvas.setStrokeColor(colors.lightgrey)
    canvas.setLineWidth(0.4)
    canvas.line(2.5*cm, 1.8*cm, w - 2.5*cm, 1.8*cm)
    canvas.restoreState()

# ── Document ──────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    str(OUTPUT),
    pagesize=A4,
    leftMargin=2.5*cm, rightMargin=2.5*cm,
    topMargin=2.8*cm, bottomMargin=2.5*cm,
    title="Chest X-Ray Pneumonia Detection — Technical Report",
    author="Suhail Saifi",
)

story = []

# ══════════════════════════════════════════════════════════════════════════════
# TITLE PAGE
# ══════════════════════════════════════════════════════════════════════════════
story.append(SP(40))
story.append(Paragraph(
    "Chest X-Ray Pneumonia Detection<br/>using ResNet50 Transfer Learning",
    title_style,
))
story.append(SP(10))
story.append(HR())
story.append(SP(10))
story.append(Paragraph("Author: <b>Suhail Saifi</b>", subtitle_style))
story.append(Paragraph("mPragati AI/Software Engineer Assessment", subtitle_style))
story.append(Paragraph("October 2026", subtitle_style))
story.append(SP(30))

# Executive summary box
exec_data = [[
    Paragraph(
        "<b>Summary</b><br/>"
        "A binary chest X-ray classifier (NORMAL vs. PNEUMONIA) built on ResNet50 transfer "
        "learning, served via a FastAPI REST API, and containerized with Docker. The model "
        "achieves <b>AUC-ROC = 0.9576</b> and <b>92% accuracy</b> at the tuned threshold of 0.87, "
        "with PNEUMONIA recall of <b>96%</b> and NORMAL recall of <b>85%</b>. "
        "Live API tests confirm correct predictions: NORMAL at 100% confidence "
        "and PNEUMONIA at 96.9% confidence.",
        body_style,
    )
]]
exec_table = Table(exec_data, colWidths=[15*cm])
exec_table.setStyle(TableStyle([
    ("BACKGROUND",   (0, 0), (-1, -1), LBLUE),
    ("BOX",          (0, 0), (-1, -1), 1.2, BLUE),
    ("TOPPADDING",   (0, 0), (-1, -1), 10),
    ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
    ("LEFTPADDING",  (0, 0), (-1, -1), 12),
    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
]))
story.append(exec_table)
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PROBLEM DEFINITION
# ══════════════════════════════════════════════════════════════════════════════
story.append(H("1. Problem Definition"))
story.append(P(
    "Pneumonia is responsible for approximately <b>2.5 million deaths per year</b> globally, "
    "making it the single largest infectious cause of death in children under five. "
    "Early and accurate diagnosis is critical, yet manual interpretation of chest X-rays "
    "is slow, subjective, and heavily dependent on radiologist availability — a resource "
    "that is scarce in many parts of the world."
))
story.append(P(
    "This project addresses the problem as a <b>binary image classification task</b>: "
    "given a posteroanterior (PA) chest X-ray image, classify it as either "
    "<b>NORMAL</b> or <b>PNEUMONIA</b>. The system is designed as an "
    "<i>AI-assisted decision-support tool</i> for radiologists and clinicians — "
    "not a replacement. It flags suspicious cases for priority review, reducing "
    "cognitive load and enabling faster triage."
))

story.append(SH("Task Specification"))
spec_data = [
    ["Parameter",     "Specification"],
    ["Intended user", "Radiologists, clinicians, emergency triage staff"],
    ["Input",         "JPEG or PNG chest X-ray (any resolution — resized to 224×224)"],
    ["Output",        "{prediction, confidence, prob_pneumonia, prob_normal, threshold_used, inference_time_ms}"],
    ["Classes",       "NORMAL | PNEUMONIA (binary)"],
    ["Threshold",     "0.87 (tuned); configurable per-request"],
]
story.append(make_table(spec_data, [5*cm, 10*cm]))
story.append(SP(8))

story.append(SH("Success Criteria"))
story.append(B("Overall accuracy > 90% on the held-out test set"))
story.append(B("PNEUMONIA recall (sensitivity) > 95% — minimising missed diagnoses"))
story.append(B("AUC-ROC > 0.95 — strong discriminative ability across all thresholds"))
story.append(SP())
story.append(P(
    "<b>Why AI?</b> The volume of chest X-rays produced in hospitals far exceeds "
    "the capacity for timely manual review. An AI system provides an instant second "
    "opinion, consistent scoring criteria, and can help prioritise which patients "
    "need urgent radiologist attention."
))
story.append(SP())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATASET & DATA STRATEGY
# ══════════════════════════════════════════════════════════════════════════════
story.append(H("2. Dataset and Data Strategy"))
story.append(P(
    "The model was trained and evaluated on the <b>Kaggle Chest X-Ray Images "
    "(Pneumonia)</b> dataset (Kermany et al., 2018). Source: "
    "https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia"
))

story.append(SH("Dataset Statistics"))
ds_data = [
    ["Split",    "NORMAL",  "PNEUMONIA", "Total"],
    ["Train",    "1,349",   "3,883",     "5,232"],
    ["Test",     "234",     "390",       "624"],
    ["Total",    "1,583",   "4,273",     "5,856"],
]
story.append(make_table(ds_data, [4*cm, 3.5*cm, 3.5*cm, 3.5*cm]))
story.append(SP(6))
story.append(P(
    "The dataset has a <b>1:2.9 class imbalance</b> (NORMAL:PNEUMONIA). "
    "The original validation split (16 images) was too small to be statistically "
    "meaningful, so it was merged back into the training set, with an 80/20 "
    "stratified split used instead."
))

story.append(SH("Preprocessing"))
story.append(B(
    "Images decoded with OpenCV (cv2.imdecode), converted from BGR to RGB, "
    "and resized to 224×224 pixels."
))
story.append(B(
    "Pixel normalisation uses ResNet50's <b>preprocess_input</b> function, "
    "which subtracts ImageNet BGR channel means [103.939, 116.779, 123.68]. "
    "This is NOT a simple /255 rescale — using the correct function is essential "
    "for transfer learning to work correctly."
))
story.append(B("No augmentation is applied to the validation or test sets."))

story.append(SH("Data Augmentation Strategy"))
story.append(P(
    "Class-specific augmentation is used to compensate for the 1:2.9 imbalance:"
))
story.append(B(
    "NORMAL (minority class): stronger augmentation — rotation ±15°, zoom 15%, "
    "width/height shifts, shear, horizontal flip."
))
story.append(B(
    "PNEUMONIA: lighter augmentation — rotation ±10°, zoom 10%, horizontal flip."
))
story.append(B(
    "Balanced batch generator yields equal batches from each class with sample "
    "weights baked in (Keras 3 does not support class_weight with Python generators)."
))
story.append(B("Data leakage prevention: test set is completely held out throughout training."))
story.append(SP())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL ARCHITECTURE & TRAINING
# ══════════════════════════════════════════════════════════════════════════════
story.append(H("3. Model Architecture and Training"))
story.append(P(
    "The model uses <b>transfer learning</b> with ResNet50 (He et al., 2015) "
    "pretrained on ImageNet as a feature extractor backbone. ResNet50 contains "
    "approximately 23.6M parameters across 50 layers with residual connections "
    "that enable very deep network training without vanishing gradients."
))

story.append(SH("Architecture"))
arch_data = [
    ["Layer",                   "Output Shape",  "Parameters"],
    ["ResNet50 backbone",        "(7, 7, 2048)",  "~23.6M (frozen phase 1)"],
    ["GlobalAveragePooling2D",   "(2048,)",       "0"],
    ["BatchNormalization",       "(2048,)",       "8,192"],
    ["Dense(256, ReLU)",         "(256,)",        "524,544"],
    ["Dropout(0.4)",             "(256,)",        "0"],
    ["Dense(64, ReLU)",          "(64,)",         "16,448"],
    ["Dropout(0.2)",             "(64,)",         "0"],
    ["Dense(1, Sigmoid)",        "(1,)",          "65"],
]
story.append(make_table(arch_data, [6.5*cm, 4*cm, 4.5*cm]))
story.append(SP(8))

story.append(SH("Two-Phase Training Strategy"))
story.append(P(
    "<b>Phase 1 — Head Training (10 epochs):</b> ResNet50 base is fully frozen. "
    "Only the custom classification head is trained with Adam at lr=1e-3. "
    "Val AUC reached <b>0.9971</b>. This phase stabilises the new layers before "
    "any disturbance to pretrained features."
))
story.append(P(
    "<b>Phase 2 — Fine-Tuning (15 epochs):</b> The top 30 layers of ResNet50 "
    "(layers 146–175) are unfrozen. The entire model is trained end-to-end with "
    "Adam at lr=1e-4 to prevent catastrophic forgetting. Val AUC reached <b>0.9913</b>."
))

story.append(SH("Training Configuration"))
train_data = [
    ["Setting",       "Value"],
    ["Framework",     "TensorFlow 2.21 / Keras 3"],
    ["Loss",          "Binary cross-entropy with sample weights"],
    ["Optimizer",     "Adam (lr=1e-3 phase 1, lr=1e-4 phase 2)"],
    ["EarlyStopping", "monitor=val_auc, patience=5, restore_best_weights=True"],
    ["ReduceLROnPlateau","factor=0.3, patience=3, monitor=val_auc"],
    ["Input size",    "224 × 224 × 3"],
    ["Batch size",    "32 (balanced: 16 NORMAL + 16 PNEUMONIA)"],
]
story.append(make_table(train_data, [6*cm, 9*cm]))
story.append(SP())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EVALUATION RESULTS
# ══════════════════════════════════════════════════════════════════════════════
story.append(H("4. Evaluation Results"))
story.append(P(
    "The model is evaluated on the held-out test set of 624 images. "
    "The AUC-ROC of <b>0.9576</b> exceeds the target of 0.95, confirming strong "
    "discriminative ability. Threshold tuning dramatically improves clinical utility "
    "compared to the default sigmoid output of 0.5."
))

story.append(SH("Table 1: Performance at Different Thresholds"))
perf_data = [
    ["Metric",                  "Threshold = 0.5 (default)", "Threshold = 0.87 (optimal)"],
    ["Accuracy",                "78%",    "92%"],
    ["AUC-ROC",                 "0.9576", "0.9576"],
    ["PNEUMONIA Precision",     "74%",    "90%"],
    ["PNEUMONIA Recall (Sens.)", "99%",   "96%"],
    ["NORMAL Precision",        "98%",    "96%"],
    ["NORMAL Recall (Spec.)",   "41%",    "85%"],
    ["F1 — PNEUMONIA",          "0.85",   "0.93"],
    ["F1 — NORMAL",             "0.58",   "0.90"],
]
story.append(make_table(perf_data, [6*cm, 4.5*cm, 4.5*cm]))
story.append(SP(8))

story.append(SH("Confusion Matrix at Threshold = 0.87 (test set, 624 images)"))
cm_data = [
    ["",                  "Predicted NORMAL", "Predicted PNEUMONIA"],
    ["Actual NORMAL",     "TN = 199",         "FP = 35"],
    ["Actual PNEUMONIA",  "FN = 16",          "TP = 374"],
]
cm_table = Table(cm_data, colWidths=[5*cm, 4.5*cm, 4.5*cm])
cm_table.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0),  BLUE),
    ("BACKGROUND",    (0, 0), (0, -1),  BLUE),
    ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
    ("TEXTCOLOR",     (0, 0), (0, -1),  WHITE),
    ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("FONTNAME",      (0, 0), (0, -1),  "Helvetica-Bold"),
    ("FONTSIZE",      (0, 0), (-1, -1), 9),
    ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ("ROWBACKGROUNDS",(1, 1), (-1, -1), [LGRAY, WHITE]),
    ("GRID",          (0, 0), (-1, -1), 0.3, colors.grey),
    ("TOPPADDING",    (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.append(cm_table)
story.append(SP(8))

story.append(SH("Live API Test Results"))
story.append(P(
    "The following results were obtained from live API testing of the deployed system:"
))
api_data = [
    ["Test",             "Prediction", "Confidence", "prob_pneumonia", "Inference Time", "Status"],
    ["NORMAL X-ray",     "NORMAL",     "100.0%",     "0.01%",         "149 ms",         "Correct"],
    ["PNEUMONIA X-ray",  "PNEUMONIA",  "96.9%",      "96.9%",         "700 ms",         "Correct"],
]
story.append(make_table(api_data, [3.5*cm, 2.5*cm, 2.5*cm, 3*cm, 2.5*cm, 1.5*cm]))
story.append(SP(6))
story.append(P(
    "The longer inference time for the PNEUMONIA case (700ms) is because the "
    "/predict/explain endpoint was used, which includes Grad-CAM heatmap generation. "
    "The fast /predict endpoint consistently achieves ~150ms."
))
story.append(P(
    "<b>Key finding:</b> Threshold tuning from 0.5 to 0.87 improved NORMAL recall "
    "(specificity) from 41% to 85% while keeping PNEUMONIA recall at 96% — "
    "a major clinical improvement with only a minor trade-off in sensitivity."
))
story.append(SP())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ERROR ANALYSIS & FAILURE CASES
# ══════════════════════════════════════════════════════════════════════════════
story.append(H("5. Error Analysis and Failure Cases"))
story.append(P(
    "Understanding failure modes is critical before any clinical deployment. "
    "At threshold 0.87, the model makes 51 errors on the 624-image test set: "
    "35 False Positives and 16 False Negatives."
))

story.append(SH("False Positives (35 cases — NORMAL predicted as PNEUMONIA)"))
story.append(P(
    "Normal chest X-rays classified as pneumonia. Common causes include subtle "
    "opacities, atelectasis, pleural effusion, or image artefacts that superficially "
    "resemble early pneumonia patterns. <b>Clinical impact:</b> unnecessary antibiotic "
    "treatment, patient anxiety, and added diagnostic costs. These are undesirable "
    "but not immediately life-threatening."
))

story.append(SH("False Negatives (16 cases — PNEUMONIA missed)"))
story.append(P(
    "Pneumonia cases missed by the model. <b>These are the most dangerous errors</b> "
    "in a clinical context. They typically involve subtle early-stage consolidation, "
    "opacities obscured by overlying structures, or atypical presentations. "
    "The threshold was specifically tuned to 0.87 (from 0.5) to minimise these cases — "
    "at threshold 0.5, the raw sigmoid output produces 41% NORMAL recall (many FPs), "
    "but raising the threshold to 0.87 dramatically reduces FPs while PNEUMONIA recall "
    "only drops from 99% to 96%."
))

story.append(SH("Grad-CAM Analysis"))
story.append(P(
    "Gradient-weighted Class Activation Mapping (Grad-CAM) was used to validate "
    "that the model attends to clinically relevant regions:"
))
story.append(B(
    "PNEUMONIA cases: activations concentrate in the lower lung lobes — the typical "
    "anatomical location of bacterial and viral pneumonia consolidation."
))
story.append(B(
    "NORMAL cases: more diffuse activation spread across both lung fields with no "
    "single dominant hot region — consistent with the absence of focal pathology."
))
story.append(B(
    "No activations on cardiac silhouette, clavicles, or image borders — "
    "confirming the model uses anatomically appropriate features."
))

story.append(SH("Known Limitations"))
story.append(B("Single dataset from one institution — may not generalise to different equipment or populations."))
story.append(B("No demographic diversity analysis performed (age, sex, ethnicity)."))
story.append(B("Trained only on bacterial and viral pneumonia; not validated for COVID-19 or fungal pneumonia."))
story.append(B("Not validated on clinical-grade PACS/DICOM images; tested only on JPEG/PNG exports."))
story.append(B("Should not be used as a standalone diagnostic tool without radiologist oversight."))
story.append(SP())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SYSTEM ARCHITECTURE & INFERENCE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
story.append(H("6. System Architecture and Inference Pipeline"))

story.append(SH("Inference Pipeline"))
pipeline_steps = [
    "Input X-Ray (JPEG/PNG)",
    "  ↓",
    "Preprocessing: cv2.imdecode → resize(224×224) → BGR2RGB → preprocess_input",
    "  ↓",
    "ResNet50 Backbone (frozen conv layers 1–145, fine-tuned layers 146–175)",
    "  ↓",
    "Custom Head: GAP → BatchNorm → Dense(256) → Dropout → Dense(64) → Dropout → Sigmoid",
    "  ↓",
    "Probability score [0.0, 1.0]",
    "  ↓",
    "Threshold (0.87): prob ≥ 0.87 → PNEUMONIA,  prob < 0.87 → NORMAL",
    "  ↓",
    "Output: {prediction, confidence, prob_pneumonia, prob_normal,",
    "         threshold_used, inference_time_ms}",
]
for step in pipeline_steps:
    story.append(C(step))
story.append(SP(6))

story.append(SH("API Endpoints"))
api_ep_data = [
    ["Method", "Endpoint",          "Description"],
    ["GET",    "/health",           "Server status and active threshold"],
    ["GET",    "/model/info",       "Model metadata (path, input size, classes)"],
    ["POST",   "/predict",          "Single image — ~150ms inference"],
    ["POST",   "/predict/explain",  "Classification + Grad-CAM heatmap (base64 PNG) — ~700ms"],
]
story.append(make_table(api_ep_data, [2*cm, 5*cm, 8*cm]))
story.append(SP(6))

story.append(SH("Infrastructure"))
story.append(B("Framework: FastAPI served via Uvicorn — async, non-blocking request handling"))
story.append(B("Containerisation: Docker + docker-compose — runs identically on any device"))
story.append(B("Model load: once at startup; TensorFlow session reused for subsequent requests"))
story.append(B("Web UI: localhost:8000 — drag-and-drop interface with Single Image and Batch tabs"))
story.append(B("Inference time: ~150ms (/predict) or ~700–900ms (/predict/explain with Grad-CAM)"))
story.append(SP())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — BONUS IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════
story.append(H("7. Bonus Implementations"))

bonus_data = [
    ["Feature",              "Description",                                            "Location"],
    ["Docker",               "Full containerisation — docker build + docker run -p 8000:8000", "Dockerfile, docker-compose.yml"],
    ["Grad-CAM",             "Heatmap overlay showing which lung regions the model attends to", "src/inference/grad_cam.py"],
    ["Web UI",               "localhost:8000 — Single Image tab (result + Grad-CAM) and Batch tab (summary + expandable cards)", "src/api/static/index.html"],
    ["ONNX Export",          "scripts/export_onnx.py exports model for edge/cross-platform deployment", "scripts/export_onnx.py"],
    ["Automated Tests",      "pytest suite covering all endpoints with real image fixtures",    "tests/test_api.py"],
    ["Threshold Tuning",     "F1-score sweep to find optimal threshold (0.87 vs default 0.5)", "scripts/evaluate.py"],
]
bonus_table = Table(bonus_data, colWidths=[3.5*cm, 7*cm, 4.5*cm])
bonus_table.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0),  BLUE),
    ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
    ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("FONTSIZE",      (0, 0), (-1, -1), 9),
    ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LGRAY, WHITE]),
    ("GRID",          (0, 0), (-1, -1), 0.3, colors.grey),
    ("TOPPADDING",    (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ("WORDWRAP",      (0, 0), (-1, -1), True),
]))
story.append(bonus_table)
story.append(SP(16))

story.append(HR())
story.append(Paragraph(
    "Chest X-Ray Pneumonia Detection — Technical Report | "
    "mPragati AI/Software Engineer Assessment | Author: Suhail Saifi | October 2026",
    footer_style,
))

# ── Build ─────────────────────────────────────────────────────────────────────
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"Report saved to: {OUTPUT}")

if __name__ == "__main__":
    pass
