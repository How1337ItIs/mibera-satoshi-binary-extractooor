"""
extractor/html_report.py
Creates a standalone self‑viewing HTML report embedding:
  • counts, recall/FP numbers
  • bar chart as embedded PNG (matplotlib)
  • confidence overlay as <img> tag
"""

from pathlib import Path
import matplotlib.pyplot as plt
import base64, io, json, datetime

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    return base64.b64encode(buf.getvalue()).decode()

def build_report(stats_dict, conf_overlay_path, out_html):
    # --- 1. bar plot of bit counts ------------------------------------------
    fig, ax = plt.subplots(figsize=(4,2))
    ax.bar(stats_dict.keys(), stats_dict.values(), color='skyblue')
    ax.set_title("Bit classification counts")
    ax.set_ylabel("cells")
    b64_plot = fig_to_b64(fig); plt.close(fig)

    # --- 2. overlay image ----------------------------------------------------
    with open(conf_overlay_path, 'rb') as f:
        b64_img = base64.b64encode(f.read()).decode()

    # --- 3. dump HTML --------------------------------------------------------
    html = f"""<!DOCTYPE html><html><head>
        <meta charset=\"utf-8\"><title>Extraction Report</title></head><body>
        <h1>Extraction Report – {datetime.datetime.utcnow().isoformat()}Z</h1>

        <h2>Key stats</h2>
        <pre>{json.dumps(stats_dict, indent=2)}</pre>

        <h2>Bit counts</h2>
        <img src=\"data:image/png;base64,{b64_plot}\" />

        <h2>Confidence overlay</h2>
        <img src=\"data:image/png;base64,{b64_img}\" style=\"max-width:100%;\" />

        </body></html>"""
    Path(out_html).write_text(html, encoding='utf-8') 