"""Test the population benchmark engine with demo users."""
import sys
sys.path.insert(0, 'src/boxbunny_dashboard')
from boxbunny_dashboard.benchmarks import BenchmarkEngine
from IPython.display import HTML, display

engine = BenchmarkEngine()

test_cases = [
    {
        "name": "Alex", "subtitle": "Male, 22, Beginner",
        "stats": {"avg_reaction_ms": 340, "punches_per_minute": 47,
                  "total_punches": 95},
        "age": 22, "gender": "male", "level": "beginner",
    },
    {
        "name": "Maria", "subtitle": "Female, 28, Intermediate",
        "stats": {"avg_reaction_ms": 280, "punches_per_minute": 65,
                  "defense_rate": 0.65},
        "age": 28, "gender": "female", "level": "intermediate",
    },
    {
        "name": "Jake", "subtitle": "Male, 31, Advanced",
        "stats": {"avg_reaction_ms": 220, "punches_per_minute": 90,
                  "defense_rate": 0.80},
        "age": 31, "gender": "male", "level": "advanced",
    },
]

all_html = ""
for tc in test_cases:
    results = engine.get_all_percentiles(
        tc["stats"], age=tc["age"],
        gender=tc["gender"], level=tc["level"],
    )

    rows_html = ""
    for metric, data in results.items():
        pct = data.get("percentile", 50)
        tier = data.get("tier", "Unknown")
        comparison = data.get("comparison", "")

        if pct >= 75:
            bar_color = "#00E676"
        elif pct >= 50:
            bar_color = "#FFC107"
        elif pct >= 25:
            bar_color = "#FF9800"
        else:
            bar_color = "#FF5722"

        metric_label = metric.replace("_", " ").title()
        rows_html += f"""
        <tr>
            <td style="padding:8px;color:#E0E0E0;font-size:13px;
                        white-space:nowrap">{metric_label}</td>
            <td style="padding:8px;width:200px">
                <div style="background:#333;border-radius:4px;
                            height:20px;position:relative">
                    <div style="background:{bar_color};
                                border-radius:4px;height:20px;
                                width:{pct}%;display:flex;
                                align-items:center;
                                justify-content:center;
                                font-size:11px;font-weight:bold;
                                color:#0D0D0D;min-width:30px">
                        {pct}%
                    </div>
                </div>
            </td>
            <td style="padding:8px;color:{bar_color};font-weight:bold;
                        font-size:13px">{tier}</td>
            <td style="padding:8px;color:#9E9E9E;font-size:12px">
                {comparison}</td>
        </tr>
        """

    all_html += f"""
    <div style="background:#1A1A1A;padding:16px;border-radius:12px;
                margin:12px 0;font-family:sans-serif">
        <h3 style="color:#00E676;margin:0 0 4px 0">{tc['name']}</h3>
        <p style="color:#9E9E9E;margin:0 0 12px 0;font-size:13px">
            {tc['subtitle']}</p>
        <table style="width:100%;border-collapse:collapse">
            <tr style="border-bottom:1px solid #333">
                <th style="text-align:left;padding:6px;color:#9E9E9E;
                            font-size:11px">METRIC</th>
                <th style="text-align:left;padding:6px;color:#9E9E9E;
                            font-size:11px">PERCENTILE</th>
                <th style="text-align:left;padding:6px;color:#9E9E9E;
                            font-size:11px">TIER</th>
                <th style="text-align:left;padding:6px;color:#9E9E9E;
                            font-size:11px">COMPARISON</th>
            </tr>
            {rows_html}
        </table>
    </div>
    """

display(HTML(f"""
<div style="max-width:800px">
    <h2 style="color:#E0E0E0;font-family:sans-serif;margin-bottom:4px">
        Population Benchmark Results</h2>
    <p style="color:#9E9E9E;font-family:sans-serif;font-size:13px;
              margin-top:0">
        Percentile rankings computed against
        age/gender/level population norms
    </p>
    {all_html}
</div>
"""))
