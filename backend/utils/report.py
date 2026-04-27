import io
from datetime import datetime
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT


DARK_BG = HexColor("#1a1a2e")
ACCENT = HexColor("#0f3460")
WARNING = HexColor("#e94560")
SUCCESS = HexColor("#4ecca3")
TEXT_DARK = HexColor("#2d2d2d")
TEXT_LIGHT = HexColor("#666666")
BORDER = HexColor("#e0e0e0")

def pil_to_rl_image(pil_img: Image.Image, width: float) -> RLImage:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    # Maintain aspect ratio
    w, h = pil_img.size
    aspect = h / w
    return RLImage(buf, width=width, height=width * aspect)


def generate_report(
    original_image: Image.Image,
    heatmap_image: Image.Image,
    prediction: dict,
    llm_explanation: str,
) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    elements = []

    # Custom styles
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        fontSize=24, textColor=ACCENT, spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=10, textColor=TEXT_LIGHT, spaceAfter=20, alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "Heading", parent=styles["Heading2"],
        fontSize=14, textColor=ACCENT, spaceBefore=16, spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, textColor=TEXT_DARK, leading=14, spaceAfter=6,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"],
        fontSize=8, textColor=TEXT_LIGHT, leading=11, spaceBefore=20,
        borderColor=BORDER, borderWidth=1, borderPadding=8,
    )

    elements.append(Paragraph("MedLens — Skin Lesion Analysis Report", title_style))
    elements.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        subtitle_style,
    ))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Uploaded Image &amp; Grad-CAM Heatmap", heading_style))

    img_width = 2.4 * inch
    orig_rl = pil_to_rl_image(original_image.resize((224, 224)), img_width)
    heat_rl = pil_to_rl_image(heatmap_image, img_width)

    img_table = Table(
        [[orig_rl, heat_rl]],
        colWidths=[img_width + 10, img_width + 10],
    )
    img_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(img_table)

    label_table = Table(
        [["Original Image", "Model Attention (Grad-CAM)"]],
        colWidths=[img_width + 10, img_width + 10],
    )
    label_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TEXTCOLOR", (0, 0), (-1, -1), TEXT_LIGHT),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    elements.append(label_table)
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Classification Results", heading_style))

    severity = prediction["severity"]
    severity_color = WARNING if "cancer" in severity else SUCCESS

    results_data = [
        ["Primary Diagnosis", prediction["class_name"]],
        ["Confidence", f"{prediction['confidence'] * 100:.1f}%"],
        ["Severity", severity],
        ["Recommended Urgency", prediction["urgency"]],
    ]

    results_table = Table(results_data, colWidths=[2.2 * inch, 3.5 * inch])
    results_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), TEXT_LIGHT),
        ("TEXTCOLOR", (1, 0), (1, -1), TEXT_DARK),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, BORDER),
    ]))
    elements.append(results_table)
    elements.append(Spacer(1, 6))
    elements.append(Paragraph("All Predictions", heading_style))

    pred_header = [["Class", "Confidence"]]
    pred_rows = [
        [prediction["all_predictions"][i][0],
         f"{prediction['all_predictions'][i][1] * 100:.1f}%"]
        for i in range(len(prediction["all_predictions"]))
    ]

    pred_table = Table(pred_header + pred_rows, colWidths=[3 * inch, 2 * inch])
    pred_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (-1, 0), ACCENT),
        ("LINEBELOW", (0, 0), (-1, 0), 1, ACCENT),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, BORDER),
    ]))
    elements.append(pred_table)
    elements.append(Paragraph("AI-Generated Explanation", heading_style))

    for para in llm_explanation.strip().split("\n"):
        para = para.strip()
        if para:
            elements.append(Paragraph(para, body_style))

    elements.append(Paragraph(
        "<b>Important Disclaimer:</b> This report was generated by MedLens, an AI-powered "
        "screening tool. It is NOT a medical diagnosis. The classification model has an "
        "accuracy of approximately 82% and may produce incorrect results. Image quality, "
        "lighting, and lesion orientation can affect accuracy. Always consult a qualified "
        "dermatologist for proper diagnosis and treatment. Do not make medical decisions "
        "based solely on this report.",
        disclaimer_style,
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer