#!/usr/bin/env python
"""
Plot GRPO metrics (entropy, reward, response length) from a JSONL log to SVG.
"""

import argparse
import json
import os


def _load_metrics(path):
    steps = []
    rewards = []
    entropies = []
    response_lengths = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = payload.get("trainer/global_step") or payload.get("step")
            reward = payload.get("train/reward") or payload.get("reward")
            entropy = payload.get("train/policy_entropy") or payload.get("policy_entropy")
            resp_len = payload.get("train/response_length") or payload.get("response_length")
            if step is None or reward is None or entropy is None or resp_len is None:
                continue
            steps.append(float(step))
            rewards.append(float(reward))
            entropies.append(float(entropy))
            response_lengths.append(float(resp_len))
    return steps, rewards, entropies, response_lengths


def _bounds(values):
    if not values:
        return 0.0, 1.0
    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    return vmin, vmax


def _polyline_points(xs, ys, x0, y0, w, h):
    if not xs:
        return ""
    xmin, xmax = _bounds(xs)
    ymin, ymax = _bounds(ys)
    pts = []
    span_x = xmax - xmin if xmax != xmin else 1.0
    span_y = ymax - ymin if ymax != ymin else 1.0
    for x, y in zip(xs, ys):
        px = x0 + (x - xmin) * (w / span_x)
        py = y0 + h - (y - ymin) * (h / span_y)
        pts.append(f"{px:.2f},{py:.2f}")
    return " ".join(pts)


def _render_panel(svg, label, xs, ys, x0, y0, w, h):
    y_min, y_max = _bounds(ys)
    svg.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#222" stroke-width="1"/>')
    svg.append(f'<text x="{x0}" y="{y0 - 8}" font-size="14" fill="#111">{label}</text>')
    svg.append(f'<text x="{x0}" y="{y0 + h + 16}" font-size="10" fill="#555">step</text>')
    svg.append(f'<text x="{x0 - 8}" y="{y0 + 10}" font-size="10" fill="#555" text-anchor="end">{y_max:.3f}</text>')
    svg.append(f'<text x="{x0 - 8}" y="{y0 + h}" font-size="10" fill="#555" text-anchor="end">{y_min:.3f}</text>')
    points = _polyline_points(xs, ys, x0, y0, w, h)
    if points:
        svg.append(f'<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{points}"/>')


def _render_svg(out_path, steps, rewards, entropies, response_lengths):
    width = 1200
    height = 800
    margin_left = 70
    margin_right = 40
    margin_top = 50
    panel_height = 200
    gap = 45
    plot_width = width - margin_left - margin_right

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="30" font-size="18" fill="#111">GRPO Metrics</text>',
    ]

    panels = [
        ("Policy Entropy", entropies),
        ("Mean Reward", rewards),
        ("Mean Response Length", response_lengths),
    ]
    for idx, (label, series) in enumerate(panels):
        y0 = margin_top + idx * (panel_height + gap)
        _render_panel(svg, label, steps, series, margin_left, y0, plot_width, panel_height)

    svg.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(svg))


def main():
    parser = argparse.ArgumentParser(description="Plot GRPO metrics JSONL to SVG.")
    parser.add_argument("metrics_path", help="Path to GRPO metrics JSONL (from GRPO_METRICS_PATH).")
    parser.add_argument("--out", default=None, help="Output SVG path (default: <metrics_path>.svg)")
    args = parser.parse_args()

    steps, rewards, entropies, response_lengths = _load_metrics(args.metrics_path)
    if not steps:
        raise SystemExit(f"No metrics found in {args.metrics_path}")
    out_path = args.out or f"{os.path.splitext(args.metrics_path)[0]}.svg"
    _render_svg(out_path, steps, rewards, entropies, response_lengths)
    print(out_path)


if __name__ == "__main__":
    main()
