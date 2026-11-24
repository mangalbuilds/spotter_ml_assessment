from __future__ import annotations

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def build_report() -> None:
    out_path = Path("reports/Spotter_ML_Assessment_Report_Mangal_Soren.pdf")
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )

    styles = getSampleStyleSheet()
    navy = colors.HexColor("#064A56")
    slate = colors.HexColor("#334155")
    light_bg = colors.HexColor("#F1F5F9")

    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18, leading=22, textColor=navy, spaceAfter=6)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12.5, leading=16, textColor=navy, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5, leading=13.5, textColor=slate, spaceAfter=5)
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=14, bulletIndent=4, spaceAfter=3)
    caption = ParagraphStyle("Cap", parent=styles["Normal"], fontSize=8.5, leading=11, textColor=colors.HexColor("#64748B"), alignment=1)

    story = []

    # Header
    story.append(Paragraph("Spotter AI — Machine Learning Engineer Assessment", h1))
    story.append(Paragraph("<b>Freight Rate Prediction Report: Data Audit, Temporal Validation & December 2025 Forecast</b><br/>Candidate: Mangal Soren | Dataset: 48,000 Dev Loads (Jan–Oct 2025) & 12,000 Validation Loads (Nov–Dec 2025)", body))
    story.append(Spacer(1, 6))

    # 1. Data-Quality Audit
    story.append(Paragraph("1. Key Exploratory Findings & Data-Quality Fixes", h2))
    findings = [
        "<b>Trap 1 — Upstream Poisoning in <font name='Courier'>quote_signal</font>:</b> Monthly correlation analysis exposed three regimes: accurate RPM proxy in Jan/Feb/Mar/Jun/Sep (+0.45 to +0.49), <i>sign-inverted</i> in Apr/May/Jul/Oct (-0.43 to -0.47), and pure white noise in Aug (-0.01) as well as all of validation Nov–Dec 2025 (corr with distance ~0.00, Dry Van = Reefer = 2.05). Moreover, <font name='Courier'>december_chart_inputs.csv</font> omits it entirely. Dropping <font name='Courier'>quote_signal</font> cut Sep–Oct holdout MAE from $156.51 down to $114.62.",
        "<b>Trap 2 — Negative & Missing Load Weights:</b> 292 training rows and 145 validation rows had sign-flipped negative weights (down to -47,500 lb), plus 300 train and 165 validation nulls. Recovering <font name='Courier'>abs(weight)</font> with explicit corruption flags and equipment-median imputation preserved valid truckload weights (5,000–47,500 lb).",
        "<b>Trap 3 — Macro Regime Shift & Missing <font name='Courier'>market_index</font>:</b> 374 train and 249 validation rows lacked <font name='Courier'>market_index</font>. Since within-day standard deviation is tiny (~0.025), missing values were imputed using same-date median <font name='Courier'>market_index</font>. Validation (Nov–Dec median ~0.92) sits below Jan–Oct median (1.06).",
        "<b>Trap 4 — Corrupted Target Outliers (~1.38%):</b> Across every month uniformly (~0.7% below $1.00/mi and ~0.7% above $4.50/mi), posted rates exhibited 0.2x–5.0x corruption. Filtering training outliers to [1.2, 3.5] $/mi and training with L1/RPM objectives prevented tree splits from distorting normal loads.",
        "<b>Trap 5 — Incomplete Schema in December Chart Inputs:</b> <font name='Courier'>december_chart_inputs.csv</font> lacks coordinates and <font name='Courier'>market_index</font>. Deterministic city coordinate lookup (Lexington: 36.99152, -84.99876; Fort Wayne: 41.31561, -85.36206) and daily December market index mapping enabled full inference.",
    ]
    for item in findings:
        story.append(Paragraph(f"• {item}", bullet))

    # 2. Validation Strategy & Split Rationale
    story.append(Paragraph("2. Train/Test Split & Temporal Validation Approach", h2))
    story.append(Paragraph(
        "Because <font name='Courier'>train_test.csv</font> spans Jan–Oct 2025 and <font name='Courier'>validation.csv</font> spans future months Nov–Dec 2025, random K-fold CV causes severe temporal data leakage (including leaking <font name='Courier'>quote_signal</font> and future seasonal states). Therefore, we enforced strict out-of-time evaluation: (a) a <b>2-month temporal holdout</b> (Train: Jan–Aug 2025, 38,477 loads -> Holdout: Sep–Oct 2025, 9,523 loads), and (b) <b>3-fold expanding-window CV</b> across Aug, Sep, and Oct 2025. All target encodings and medians were fit strictly inside each training fold.",
        body,
    ))

    # 3. Model Selection & Quantitative Results
    story.append(Paragraph("3. Model Reasoning & Temporal Validation Results", h2))
    story.append(Paragraph(
        "Freight rates scale linearly with distance while variance grows with haul length (heteroscedasticity). Predicting <b>rate-per-mile (<font name='Courier'>RPM = posted_rate / distance</font>)</b> and multiplying back by distance outperformed direct rate prediction ($61.16 vs $73.09 clean MAE). Our final model blends four complementary tree regressors trained on clean RPM (<b>45% HistGB L1 + 25% HistGB L2 + 15% ExtraTrees + 15% RandomForest</b>) using 31 engineered spatial, geometry (Haversine & circuity ratio), calendar, and smoothed lane target-encoded features.",
        body,
    ))

    cv_table_data = [
        ["Evaluation Fold / Model", "ALL MAE", "ALL R²", "Clean (98.6%) MAE", "Clean RMSE", "Clean MAPE", "Clean R²"],
        ["Naive HistGB WITH quote_signal (Sep–Oct)", "$156.51", "—", "$102.90", "—", "—", "—"],
        ["Ridge Baseline (Sep–Oct Holdout)", "$200.50", "0.8088", "$147.46", "$231.09", "7.03%", "0.9720"],
        ["Expanding CV Fold 1: Aug 2025 (Ensemble)", "$84.65", "0.8267", "$34.41", "$49.65", "1.51%", "0.9986"],
        ["Expanding CV Fold 2: Sep 2025 (Ensemble)", "$112.37", "0.8350", "$61.29", "$91.07", "2.53%", "0.9957"],
        ["Expanding CV Fold 3: Oct 2025 (Ensemble)", "$103.05", "0.8215", "$42.52", "$63.22", "1.80%", "0.9979"],
        ["Final 4-Model Ensemble (Sep–Oct Holdout)", "$114.62", "0.8266", "$58.84", "$84.12", "2.47%", "0.9963"],
    ]
    tbl = Table(cv_table_data, colWidths=[2.6 * inch, 0.72 * inch, 0.65 * inch, 1.05 * inch, 0.82 * inch, 0.72 * inch, 0.72 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("BACKGROUND", (0, -1), (-1, -1), light_bg),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(Spacer(1, 4))
    story.append(tbl)

    story.append(PageBreak())

    # Page 2: Visual Diagnostics & Fixed December Chart
    story.append(Paragraph("4. Diagnostic Plots: Upstream Feature Trap & Holdout Fit", h2))
    diag_img = Image("reports/eda_and_validation_diagnostics.png", width=7.1 * inch, height=2.55 * inch)
    story.append(diag_img)
    story.append(Paragraph("Left: Monthly correlation of quote_signal with RPM showing sign inversions and Aug collapse. Right: Sep–Oct Holdout predicted vs actual rates (Clean R² = 0.9963).", caption))
    story.append(Spacer(1, 10))

    story.append(Paragraph("5. Fixed December 2025 Prediction Chart (Generated by score.py)", h2))
    story.append(Paragraph(
        "On the fixed test corridor (<b>Lexington → Fort Wayne | 360 mi | Dry Van | 32,000 lb</b>), predicted rates range from <b>$829.15</b> on Sunday troughs to a Christmas week peak of <b>$844.70</b> on Dec 25 (mean <b>$837.07</b>), aligning closely with historical 2025 observations on the same lane (~$768–$810) and capturing mid-week capacity tightening plus late-December holiday demand.",
        body,
    ))
    dec_img = Image("scorer_results/candidate_december.png", width=7.1 * inch, height=3.1 * inch)
    story.append(dec_img)
    story.append(Paragraph("Official Spotter AI score.py output chart for December 1–31, 2025 (31 validated daily forecasts; 12,000 validation loads validated).", caption))

    doc.build(story)
    print(f"Generated PDF report: {out_path}")


if __name__ == "__main__":
    build_report()
