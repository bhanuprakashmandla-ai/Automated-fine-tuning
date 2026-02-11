"""HTML reporting helpers for experiment summaries."""

from __future__ import annotations

from html import escape
import json
from typing import Any, Dict, Iterable, List


def _to_pretty_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _svg_line_chart(series: Dict[str, List[float]], width: int = 640, height: int = 180) -> str:
    if not series:
        return "<p>No training logs available.</p>"

    all_values = [v for values in series.values() for v in values]
    if not all_values:
        return "<p>No training logs available.</p>"

    min_val = min(all_values)
    max_val = max(all_values)
    span = max(max_val - min_val, 1e-8)
    padding = 10
    plot_width = width - 2 * padding
    plot_height = height - 2 * padding

    def scale_points(values: List[float]) -> str:
        if len(values) == 1:
            x = padding + plot_width / 2
            y = padding + plot_height * (1 - (values[0] - min_val) / span)
            return f"M {x:.2f} {y:.2f}"
        points = []
        for idx, value in enumerate(values):
            x = padding + (plot_width * idx / (len(values) - 1))
            y = padding + plot_height * (1 - (value - min_val) / span)
            points.append((x, y))
        return "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in points)

    colors = {
        "train_loss": "#1f77b4",
        "eval_loss": "#ff7f0e",
    }

    paths = []
    for key, values in series.items():
        if not values:
            continue
        path = scale_points(values)
        color = colors.get(key, "#2ca02c")
        paths.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" />')

    legend_items = []
    for key, color in colors.items():
        if key in series:
            legend_items.append(
                f'<span style="display:inline-flex;align-items:center;margin-right:12px;">'
                f'<span style="width:12px;height:12px;background:{color};display:inline-block;margin-right:6px;"></span>'
                f'{escape(key)}'
                f'</span>'
            )

    legend = "".join(legend_items)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff" stroke="#ddd" />'
        + "".join(paths)
        + "</svg>"
        f"<div style=\"margin-top:6px;font-size:12px;color:#555;\">{legend}</div>"
    )


def _table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    header_html = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    row_html = []
    for row in rows:
        row_html.append(
            "<tr>" + "".join(f"<td>{escape(str(item))}</td>" for item in row) + "</tr>"
        )
    body_html = "".join(row_html) if row_html else "<tr><td colspan=\"6\">No data</td></tr>"
    return (
        "<table>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table>"
    )


def _svg_metric_bars(
    series: Dict[str, List[float]],
    labels: List[str],
    width: int = 640,
    height: int = 220,
) -> str:
    metrics = list(series.keys())
    if not metrics or not labels:
        return "<p>No metrics available.</p>"

    max_val = max((max(values) for values in series.values() if values), default=0.0)
    if max_val <= 0:
        max_val = 1.0

    padding = 40
    plot_width = width - 2 * padding
    plot_height = height - 2 * padding
    group_count = len(labels)
    metric_count = len(metrics)
    group_width = plot_width / max(group_count, 1)
    bar_width = group_width / max(metric_count, 1) * 0.7
    bar_gap = (group_width - bar_width * metric_count) / 2

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    bars = []
    for group_idx, label in enumerate(labels):
        x_start = padding + group_idx * group_width + bar_gap
        for metric_idx, metric in enumerate(metrics):
            values = series.get(metric, [])
            value = values[group_idx] if group_idx < len(values) else 0
            scaled_height = (value / max_val) * plot_height
            x = x_start + metric_idx * bar_width
            y = padding + (plot_height - scaled_height)
            color = colors[metric_idx % len(colors)]
            bars.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{scaled_height:.2f}" '
                f'fill="{color}"><title>{escape(metric)}: {value:.4f}</title></rect>'
            )

        label_x = padding + group_idx * group_width + group_width / 2
        label_y = padding + plot_height + 16
        bars.append(
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="middle" font-size="11">'
            f'{escape(label)}</text>'
        )

    legend_items = []
    for metric_idx, metric in enumerate(metrics):
        color = colors[metric_idx % len(colors)]
        legend_items.append(
            f'<span style="display:inline-flex;align-items:center;margin-right:12px;">'
            f'<span style="width:12px;height:12px;background:{color};display:inline-block;margin-right:6px;"></span>'
            f'{escape(metric)}'
            f'</span>'
        )

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff" stroke="#ddd" />'
        + "".join(bars)
        + "</svg>"
        f"<div style=\"margin-top:6px;font-size:12px;color:#555;\">{''.join(legend_items)}</div>"
    )


def generate_html_report(
    results: Dict[str, Any],
    output_dir: str,
    dataset_name: str,
    dataset_label: str,
) -> str:
    exp_results = results["exp_results"]
    config = results["config"]

    metrics_rows = []
    metric_series = {"f1": [], "precision": [], "recall": [], "latency": []}
    metric_labels = []
    for exp_name, exp_data in exp_results.items():
        metrics = exp_data.get("metrics", {})
        metric_labels.append(exp_name)
        for key in metric_series:
            metric_series[key].append(float(metrics.get(key, 0) or 0))
        metrics_rows.append(
            [
                exp_name,
                metrics.get("precision", "n/a"),
                metrics.get("recall", "n/a"),
                metrics.get("f1", "n/a"),
                metrics.get("latency", "n/a"),
            ]
        )

    exp_sections = []
    for exp_name, exp_data in exp_results.items():
        logs = exp_data.get("logs")
        config_data = exp_data.get("config", {})
        series = {}
        if logs is not None and not logs.empty:
            if "train_loss" in logs.columns:
                series["train_loss"] = logs["train_loss"].dropna().tolist()
            if "eval_loss" in logs.columns:
                series["eval_loss"] = logs["eval_loss"].dropna().tolist()

        exp_sections.append(
            f"""
            <section class="card">
              <h3>Experiment: {escape(exp_name)}</h3>
              <div class="grid">
                <div>
                  <h4>Configuration</h4>
                  <pre>{escape(_to_pretty_json(config_data))}</pre>
                </div>
                <div>
                  <h4>Training Curves</h4>
                  {_svg_line_chart(series)}
                </div>
              </div>
            </section>
            """
        )

    html = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <title>Fine-tuning Report - {escape(dataset_name)}</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
        h1, h2, h3 {{ margin-bottom: 8px; }}
        .meta {{ color: #555; font-size: 14px; }}
        .card {{ border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f7f7f7; }}
        pre {{ background: #fafafa; border: 1px solid #eee; padding: 12px; overflow-x: auto; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }}
      </style>
    </head>
    <body>
      <h1>Fine-tuning Report</h1>
      <p class="meta">Dataset: {escape(dataset_name)} | Output: {escape(output_dir)} | Report ID: {escape(dataset_label)}</p>

      <section class="card">
        <h2>Experiment Summary</h2>
        <p><strong>Model:</strong> {escape(config.get("model_name", "n/a"))}</p>
        <p><strong>Chat Template:</strong> {escape(config.get("chat_template", "n/a"))}</p>
        <h3>Metrics Overview</h3>
        {_svg_metric_bars(metric_series, metric_labels)}
        {_table(["Experiment", "Precision", "Recall", "F1", "Latency"], metrics_rows)}
      </section>

      {''.join(exp_sections)}
    </body>
    </html>
    """
    return html
