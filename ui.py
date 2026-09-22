#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ui.py -- Gradio Clinical Web Interface

Two-level cancer screening interface with multi-model selector,
model comparison panel, and bilingual (EN/TH) clinical output.
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from config import (
    CLASS_CODES,
    CLASS_NAMES_EN,
    CLASS_NAMES_TH,
    CLASS_DESCRIPTIONS_EN,
    CLASS_DESCRIPTIONS_TH,
    CANCER_INDICES,
    BENIGN_INDICES,
    MODEL_REGISTRY,
    MODEL_NAMES,
    NUM_CLASSES,
)
from data import val_test_transform
from models import build_model
from evaluate import load_all_metrics


# =============================================================================
# CSS Stylesheet
# =============================================================================

GRADIO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sarabun:wght@300;400;500;600;700&display=swap');

:root {
    --slate-900: #0f172a;
    --slate-800: #1e293b;
    --slate-700: #334155;
    --slate-600: #475569;
    --slate-500: #64748b;
    --slate-400: #94a3b8;
    --slate-300: #cbd5e1;
    --slate-200: #e2e8f0;
    --slate-100: #f1f5f9;
    --slate-50:  #f8fafc;
    --red-700:   #b91c1c;
    --red-600:   #dc2626;
    --red-100:   #fee2e2;
    --red-50:    #fef2f2;
    --green-700: #15803d;
    --green-600: #16a34a;
    --green-100: #dcfce7;
    --green-50:  #f0fdf4;
}

* {
    font-family: 'Inter', 'Sarabun', -apple-system, sans-serif !important;
}

body, .gradio-container {
    background-color: var(--slate-50) !important;
}

.gradio-container {
    max-width: 1020px !important;
    margin: 0 auto !important;
}

.system-header {
    background: var(--slate-900);
    color: #ffffff;
    padding: 20px 28px;
    border-radius: 8px;
    margin-bottom: 20px;
    border-bottom: 3px solid var(--slate-700);
}

.system-header h1 {
    font-size: 17px;
    font-weight: 600;
    margin: 0 0 4px 0;
    letter-spacing: 0.3px;
    color: #ffffff;
}

.system-header .subtitle {
    font-size: 12px;
    color: var(--slate-400);
    margin: 0;
    font-weight: 400;
}

.notice-bar {
    background: var(--slate-100);
    border: 1px solid var(--slate-200);
    border-left: 3px solid var(--slate-500);
    border-radius: 4px;
    padding: 10px 14px;
    margin-bottom: 16px;
    font-size: 11px;
    color: var(--slate-600);
    line-height: 1.6;
}

.notice-bar strong {
    color: var(--slate-700);
}

.info-panel {
    background: #ffffff;
    border: 1px solid var(--slate-200);
    border-radius: 6px;
    margin-top: 14px;
    overflow: hidden;
}

.info-panel .panel-header {
    background: var(--slate-100);
    padding: 8px 14px;
    font-size: 11px;
    font-weight: 600;
    color: var(--slate-700);
    text-transform: uppercase;
    letter-spacing: 0.4px;
    border-bottom: 1px solid var(--slate-200);
}

.info-panel .panel-body {
    padding: 10px 14px;
}

.info-panel .class-row {
    display: flex;
    justify-content: space-between;
    padding: 3px 0;
    font-size: 11px;
    color: var(--slate-600);
    border-bottom: 1px solid var(--slate-100);
}

.info-panel .class-row:last-child {
    border-bottom: none;
}

.info-panel .class-row .code {
    font-weight: 600;
    color: var(--slate-800);
    min-width: 42px;
}

.class-group-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 6px 0 3px 0;
}

.class-group-label.cancer {
    color: var(--red-700);
    border-top: 1px solid var(--red-100);
    margin-top: 4px;
}

.class-group-label.benign {
    color: var(--green-700);
    border-top: 1px solid var(--green-100);
    margin-top: 4px;
}
"""


# =============================================================================
# HTML Builders
# =============================================================================

def _placeholder_html() -> str:
    """Return placeholder HTML for initial state."""
    return (
        "<div style='display:flex;align-items:center;justify-content:center;"
        "height:300px;color:#94a3b8;font-size:13px;font-family:Inter,sans-serif;"
        "text-align:center;line-height:1.8;'>"
        "Upload a dermoscopic image<br>and press Analyze to begin.</div>"
    )


def _build_comparison_html() -> str:
    """Build HTML table comparing all evaluated models."""
    all_metrics = load_all_metrics()
    if not all_metrics:
        return (
            "<div style='padding:14px;font-size:12px;color:#94a3b8;"
            "font-family:Inter,sans-serif;text-align:center;'>"
            "No model metrics available. Train models first.</div>"
        )

    F = "font-family:'Inter','Sarabun',sans-serif;"

    html = f"""<div style="{F}font-size:12px;color:#0f172a;">
    <table style="width:100%;border-collapse:collapse;font-size:11px;">
        <thead>
            <tr style="background:#f1f5f9;border-bottom:2px solid #e2e8f0;">
                <th style="text-align:left;padding:8px 10px;color:#334155;
                    font-weight:600;">Metric</th>"""

    names = list(all_metrics.keys())
    for name in names:
        html += f"""<th style="text-align:center;padding:8px 10px;color:#334155;
            font-weight:600;">{name}</th>"""
    html += "</tr></thead><tbody>"

    # Metric definitions: (display_label, key, higher_is_better, format_as_pct)
    metrics_def = [
        ("Accuracy", "overall_accuracy", True, True),
        ("Macro F1", "macro_f1", True, True),
        ("Cancer Sensitivity", "cancer_sensitivity", True, True),
        ("Specificity", "cancer_specificity", True, True),
        ("False Negative Rate", "false_negative_rate", False, True),
        ("Parameters", "total_params", None, False),
    ]

    for i, (label, key, higher_better, as_pct) in enumerate(metrics_def):
        bg = "#ffffff" if i % 2 == 0 else "#f8fafc"
        values = [all_metrics[n].get(key, 0) for n in names]

        # Determine best
        if higher_better is True:
            best_val = max(values)
        elif higher_better is False:
            best_val = min(values)
        else:
            best_val = None

        html += f'<tr style="background:{bg};border-bottom:1px solid #f1f5f9;">'
        html += f'<td style="padding:6px 10px;color:#475569;font-weight:500;">{label}</td>'

        for val in values:
            if as_pct:
                text = f"{val * 100:.1f}%"
            else:
                text = f"{val / 1e6:.1f}M"

            is_best = best_val is not None and val == best_val
            weight = "700" if is_best else "400"
            color = "#0f172a" if is_best else "#64748b"
            html += f'<td style="text-align:center;padding:6px 10px;font-weight:{weight};color:{color};">{text}</td>'

        html += "</tr>"

    html += "</tbody></table></div>"
    return html


def _build_result_html(
    model_name: str,
    is_high_risk: bool,
    cancer_prob: float,
    benign_prob: float,
    cancer_subtypes: list,
    benign_subtypes: list,
    top_idx: int,
    top_prob: float,
) -> str:
    """Build complete two-level analysis result HTML."""

    F = "font-family:'Inter','Sarabun',sans-serif;"

    if is_high_risk:
        risk_label = "HIGH RISK"
        risk_bg = "#b91c1c"
        risk_border = "#991b1b"
        banner_bg = "#fef2f2"
        banner_border = "#fecaca"
    else:
        risk_label = "LOW RISK"
        risk_bg = "#15803d"
        risk_border = "#166534"
        banner_bg = "#f0fdf4"
        banner_border = "#bbf7d0"

    # ---- 1. SCREENING RESULT ----
    html = f"""<div style="{F}font-size:13px;color:#0f172a;">

    <div style="background:{banner_bg};border:1px solid {banner_border};border-radius:6px;
        padding:18px 22px;margin-bottom:16px;">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
            <span style="font-size:11px;font-weight:600;color:#475569;text-transform:uppercase;
                letter-spacing:0.6px;">Screening Result</span>
            <span style="display:inline-block;background:{risk_bg};border:1px solid {risk_border};
                color:#ffffff;padding:4px 14px;border-radius:3px;font-size:11px;
                font-weight:600;letter-spacing:0.8px;">{risk_label}</span>
        </div>
        <div style="font-size:10px;color:#64748b;margin-bottom:12px;">Analyzed with: {model_name}</div>
        <div style="display:flex;gap:20px;">
            <div style="flex:1;">
                <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                    <span style="font-size:11px;font-weight:500;color:#64748b;">Cancer / Pre-Cancer</span>
                    <span style="font-size:11px;font-weight:600;color:#b91c1c;">{cancer_prob*100:.1f}%</span>
                </div>
                <div style="height:5px;background:#e2e8f0;border-radius:2px;overflow:hidden;">
                    <div style="height:100%;width:{cancer_prob*100:.1f}%;background:#dc2626;
                        border-radius:2px;"></div>
                </div>
            </div>
            <div style="flex:1;">
                <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                    <span style="font-size:11px;font-weight:500;color:#64748b;">Benign</span>
                    <span style="font-size:11px;font-weight:600;color:#15803d;">{benign_prob*100:.1f}%</span>
                </div>
                <div style="height:5px;background:#e2e8f0;border-radius:2px;overflow:hidden;">
                    <div style="height:100%;width:{benign_prob*100:.1f}%;background:#16a34a;
                        border-radius:2px;"></div>
                </div>
            </div>
        </div>
    </div>
    """

    # ---- 2. PRIMARY DIAGNOSIS ----
    is_cancer_class = top_idx in CANCER_INDICES
    diag_accent = "#b91c1c" if is_cancer_class else "#15803d"

    html += f"""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:6px;
        padding:16px 20px;margin-bottom:16px;border-left:3px solid {diag_accent};">
        <div style="font-size:10px;font-weight:600;color:#94a3b8;text-transform:uppercase;
            letter-spacing:0.6px;margin-bottom:8px;">Primary Diagnosis</div>
        <div style="font-size:15px;font-weight:600;color:#0f172a;margin-bottom:2px;">
            {CLASS_NAMES_EN[top_idx]}</div>
        <div style="font-size:12px;color:#475569;margin-bottom:8px;">
            {CLASS_NAMES_TH[top_idx]}</div>
        <div style="display:inline-block;background:#f1f5f9;border:1px solid #e2e8f0;
            border-radius:3px;padding:2px 10px;font-size:11px;font-weight:600;
            color:#334155;margin-bottom:10px;">Confidence: {top_prob*100:.1f}%</div>
        <div style="font-size:11px;color:#64748b;line-height:1.6;margin-top:6px;">
            {CLASS_DESCRIPTIONS_EN[top_idx]}</div>
        <div style="font-size:11px;color:#64748b;line-height:1.6;margin-top:4px;">
            {CLASS_DESCRIPTIONS_TH[top_idx]}</div>
    </div>
    """

    # ---- 3. DIFFERENTIAL ANALYSIS TABLE ----
    html += """
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:6px;
        overflow:hidden;margin-bottom:16px;">
        <div style="background:#f1f5f9;padding:9px 18px;border-bottom:1px solid #e2e8f0;">
            <span style="font-size:11px;font-weight:600;color:#334155;text-transform:uppercase;
                letter-spacing:0.5px;">Differential Probability Analysis</span>
        </div>
        <div style="padding:14px 18px;">
    """

    # Cancer subtypes
    html += """<div style="font-size:10px;font-weight:600;color:#b91c1c;text-transform:uppercase;
        letter-spacing:0.5px;padding-bottom:6px;border-bottom:1px solid #fee2e2;
        margin-bottom:8px;">Cancer / Pre-Cancer</div>"""

    for code, name_en, name_th, prob in cancer_subtypes:
        bar_w = max(prob * 100, 0.3)
        html += f"""
        <div style="display:flex;align-items:center;gap:10px;padding:5px 0;
            border-bottom:1px solid #f8fafc;">
            <span style="font-size:11px;font-weight:600;color:#0f172a;
                min-width:36px;">{code}</span>
            <span style="font-size:11px;color:#475569;flex:1;
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{name_en}</span>
            <div style="width:100px;height:4px;background:#fee2e2;border-radius:2px;
                overflow:hidden;flex-shrink:0;">
                <div style="height:100%;width:{bar_w:.1f}%;background:#dc2626;
                    border-radius:2px;"></div>
            </div>
            <span style="font-size:11px;font-weight:600;color:#b91c1c;
                min-width:44px;text-align:right;">{prob*100:.1f}%</span>
        </div>"""

    # Benign subtypes
    html += """<div style="font-size:10px;font-weight:600;color:#15803d;text-transform:uppercase;
        letter-spacing:0.5px;padding-bottom:6px;border-bottom:1px solid #dcfce7;
        margin-top:14px;margin-bottom:8px;">Benign</div>"""

    for code, name_en, name_th, prob in benign_subtypes:
        bar_w = max(prob * 100, 0.3)
        html += f"""
        <div style="display:flex;align-items:center;gap:10px;padding:5px 0;
            border-bottom:1px solid #f8fafc;">
            <span style="font-size:11px;font-weight:600;color:#0f172a;
                min-width:36px;">{code}</span>
            <span style="font-size:11px;color:#475569;flex:1;
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{name_en}</span>
            <div style="width:100px;height:4px;background:#dcfce7;border-radius:2px;
                overflow:hidden;flex-shrink:0;">
                <div style="height:100%;width:{bar_w:.1f}%;background:#16a34a;
                    border-radius:2px;"></div>
            </div>
            <span style="font-size:11px;font-weight:600;color:#15803d;
                min-width:44px;text-align:right;">{prob*100:.1f}%</span>
        </div>"""

    html += """
        </div>
    </div>
    """

    # ---- 4. CLINICAL RECOMMENDATION ----
    if is_high_risk:
        rec_border = "#fecaca"
        rec_bg = "#fef2f2"
        rec_accent = "#b91c1c"
        rec_en = (
            "Elevated probability for malignant or pre-cancerous lesion detected. "
            "Referral to a dermatologist for clinical evaluation and possible biopsy "
            "is strongly recommended."
        )
        rec_th = (
            "ตรวจพบความเสี่ยงสูงต่อรอยโรคมะเร็งหรือก่อนมะเร็ง "
            "แนะนำให้ส่งต่อแพทย์ผิวหนังเพื่อประเมินทางคลินิก "
            "และพิจารณาตรวจชิ้นเนื้อยืนยันผลทางพยาธิวิทยา"
        )
    else:
        rec_border = "#bbf7d0"
        rec_bg = "#f0fdf4"
        rec_accent = "#15803d"
        rec_en = (
            "Analysis suggests a benign lesion with low malignancy probability. "
            "Periodic self-monitoring is recommended. Consult a physician if changes "
            "in size, color, border, or symptoms are observed."
        )
        rec_th = (
            "ผลการวิเคราะห์บ่งชี้ว่าเป็นรอยโรคที่ไม่ร้ายแรง มีความเสี่ยงต่ำต่อมะเร็ง "
            "แนะนำให้ติดตามสังเกตอาการเป็นระยะ หากพบการเปลี่ยนแปลงของขนาด สี ขอบ "
            "หรือมีอาการผิดปกติ ควรปรึกษาแพทย์"
        )

    html += f"""
    <div style="background:{rec_bg};border:1px solid {rec_border};border-left:3px solid {rec_accent};
        border-radius:4px;padding:14px 18px;margin-bottom:14px;">
        <div style="font-size:10px;font-weight:600;color:{rec_accent};text-transform:uppercase;
            letter-spacing:0.5px;margin-bottom:6px;">Recommendation</div>
        <div style="font-size:11px;color:#334155;line-height:1.7;margin-bottom:4px;">{rec_en}</div>
        <div style="font-size:11px;color:#475569;line-height:1.7;">{rec_th}</div>
    </div>
    """

    # ---- 5. DISCLAIMER ----
    html += """
    <div style="padding:8px 0;">
        <div style="font-size:9px;color:#94a3b8;line-height:1.5;text-align:center;">
            This system is intended for educational and research purposes only (CPE310).
            Results are probabilistic estimates and do not constitute medical diagnosis.
            Consult qualified healthcare professionals for all clinical decisions.
        </div>
    </div>

    </div>"""

    return html


# =============================================================================
# Gradio Application
# =============================================================================

def launch_gradio_demo(
    initial_model_name: str,
    initial_model: nn.Module,
    device: torch.device,
) -> None:
    """Launch Gradio web interface with model selector and comparison panel."""
    try:
        import gradio as gr
    except ImportError:
        print("[ERROR] Gradio is not installed. Run: pip install gradio")
        return

    # Mutable state: currently loaded model
    state = {
        "model_name": initial_model_name,
        "model": initial_model,
    }
    state["model"].eval()

    def _get_available_models() -> list:
        """Return list of model names that have a trained checkpoint."""
        available = []
        all_metrics = load_all_metrics()
        for name in MODEL_NAMES:
            checkpoint = MODEL_REGISTRY[name]["checkpoint"]
            if checkpoint.exists():
                metrics = all_metrics.get(name, {})
                acc = metrics.get("overall_accuracy", 0)
                f1 = metrics.get("macro_f1", 0)
                if acc > 0:
                    label = f"{name} (Acc: {acc*100:.1f}%, F1: {f1*100:.1f}%)"
                else:
                    label = name
                available.append(label)
        return available if available else [initial_model_name]

    def _parse_model_name(label: str) -> str:
        """Extract model name from dropdown label like 'ResNet50 (Acc: 84.2%, ...)'."""
        return label.split(" (")[0].strip()

    def on_model_change(selected_label):
        """Handle model selection change."""
        model_name = _parse_model_name(selected_label)
        if model_name == state["model_name"]:
            return f"Current model: {model_name}"

        checkpoint = MODEL_REGISTRY.get(model_name, {}).get("checkpoint")
        if checkpoint is None or not checkpoint.exists():
            return f"[ERROR] Checkpoint not found for {model_name}"

        print(f"[INFO] Switching to model: {model_name}")
        new_model = build_model(model_name, device)
        new_model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        new_model.eval()

        state["model_name"] = model_name
        state["model"] = new_model
        return f"Active model: {model_name}"

    def predict(input_image):
        """Process uploaded image and return two-level classification result."""
        if input_image is None:
            return _placeholder_html()

        try:
            if not isinstance(input_image, Image.Image):
                input_image = Image.fromarray(input_image)
            img = input_image.convert("RGB")
            img_tensor = val_test_transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = state["model"](img_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            cancer_prob = float(sum(probs[i] for i in CANCER_INDICES))
            benign_prob = float(sum(probs[i] for i in BENIGN_INDICES))
            is_high_risk = cancer_prob >= 0.5

            cancer_subtypes = [
                (CLASS_CODES[i], CLASS_NAMES_EN[i], CLASS_NAMES_TH[i], float(probs[i]))
                for i in sorted(CANCER_INDICES)
            ]
            benign_subtypes = [
                (CLASS_CODES[i], CLASS_NAMES_EN[i], CLASS_NAMES_TH[i], float(probs[i]))
                for i in sorted(BENIGN_INDICES)
            ]

            cancer_subtypes.sort(key=lambda x: x[3], reverse=True)
            benign_subtypes.sort(key=lambda x: x[3], reverse=True)

            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])

            return _build_result_html(
                state["model_name"],
                is_high_risk, cancer_prob, benign_prob,
                cancer_subtypes, benign_subtypes,
                top_idx, top_prob,
            )

        except Exception as e:
            return (
                f"<div style='padding:16px;color:#b91c1c;font-size:13px;"
                f"font-family:Inter,sans-serif;'>Analysis Error: {str(e)}</div>"
            )

    # ---- BUILD GRADIO INTERFACE ----
    with gr.Blocks(css=GRADIO_CSS, title="Skin Cancer Screening System") as demo:

        gr.HTML("""
        <div class="system-header">
            <h1>Skin Cancer Classification & Screening System</h1>
            <p class="subtitle">ISIC 2019 Dataset  |  8-Class Multi-type Classification  |  Multi-Model Comparison</p>
        </div>
        """)

        gr.HTML("""
        <div class="notice-bar">
            <strong>Notice:</strong>
            This system is an AI-assisted clinical screening prototype developed for academic purposes
            (CPE310 Healthcare AI System). All results are probabilistic and must not replace
            professional dermatological evaluation and histopathological diagnosis.
        </div>
        """)

        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                # Model selector dropdown
                available_models = _get_available_models()
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0] if available_models else initial_model_name,
                    label="Select Model",
                    interactive=True,
                )
                model_status = gr.Textbox(
                    value=f"Active model: {initial_model_name}",
                    label="Model Status",
                    interactive=False,
                    max_lines=1,
                )

                input_image = gr.Image(
                    type="pil",
                    label="Dermoscopic Image",
                    height=300,
                )
                submit_btn = gr.Button(
                    "Analyze",
                    variant="primary",
                    size="lg",
                )

                gr.HTML("""
                <div class="info-panel">
                    <div class="panel-header">Detectable Conditions</div>
                    <div class="panel-body">
                        <div class="class-group-label cancer">Cancer / Pre-Cancer</div>
                        <div class="class-row"><span class="code">MEL</span><span>Melanoma</span></div>
                        <div class="class-row"><span class="code">BCC</span><span>Basal Cell Carcinoma</span></div>
                        <div class="class-row"><span class="code">SCC</span><span>Squamous Cell Carcinoma</span></div>
                        <div class="class-row"><span class="code">AK</span><span>Actinic Keratoses</span></div>
                        <div class="class-group-label benign">Benign</div>
                        <div class="class-row"><span class="code">NV</span><span>Melanocytic Nevi</span></div>
                        <div class="class-row"><span class="code">BKL</span><span>Benign Keratosis</span></div>
                        <div class="class-row"><span class="code">VASC</span><span>Vascular Lesions</span></div>
                        <div class="class-row"><span class="code">DF</span><span>Dermatofibroma</span></div>
                    </div>
                </div>
                """)

                # Model comparison panel
                gr.HTML('<div class="info-panel"><div class="panel-header">Model Comparison</div><div class="panel-body">')
                comparison_html = gr.HTML(value=_build_comparison_html())
                gr.HTML('</div></div>')

            with gr.Column(scale=3):
                result_output = gr.HTML(
                    value=_placeholder_html(),
                    label="Analysis Result",
                )

        # Event bindings
        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[model_status],
        )
        submit_btn.click(
            fn=predict,
            inputs=[input_image],
            outputs=[result_output],
        )

    print("[INFO] Launching Gradio Clinical Demo...")
    print("[INFO] Access the interface at: http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
