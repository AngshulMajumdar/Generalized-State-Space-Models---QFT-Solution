
use anyhow::Result;
use std::fs;
use std::path::Path;

#[derive(Clone, Copy)]
pub enum AxisScale {
    Linear,
    Log10,
}

pub struct Curve {
    pub label: String,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub sd: Option<Vec<f64>>,
}

pub struct PlotSpec {
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub x_scale: AxisScale,
    pub y_scale: AxisScale,
    pub curves: Vec<Curve>,
}

fn tx(v: f64, scale: AxisScale) -> f64 {
    match scale {
        AxisScale::Linear => v,
        AxisScale::Log10 => v.max(1e-300).log10(),
    }
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

pub fn write_svg(path: &Path, spec: &PlotSpec) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let w = 1000.0;
    let h = 650.0;
    let ml = 95.0;
    let mr = 250.0;
    let mt = 65.0;
    let mb = 85.0;
    let pw = w - ml - mr;
    let ph = h - mt - mb;

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for c in &spec.curves {
        for (&x, &y) in c.x.iter().zip(c.y.iter()) {
            let x_ok = x.is_finite() && (!matches!(spec.x_scale, AxisScale::Log10) || x > 0.0);
            let y_ok = y.is_finite() && (!matches!(spec.y_scale, AxisScale::Log10) || y > 0.0);
            if x_ok && y_ok {
                xs.push(tx(x, spec.x_scale));
                ys.push(tx(y, spec.y_scale));
            }
        }
        if let Some(sd) = &c.sd {
            for ((&x, &y), &s) in c.x.iter().zip(c.y.iter()).zip(sd.iter()) {
                let lo = (y - s).max(1e-300);
                let hi = (y + s).max(1e-300);
                xs.push(tx(x, spec.x_scale));
                ys.push(tx(lo, spec.y_scale));
                ys.push(tx(hi, spec.y_scale));
            }
        }
    }
    let xmin = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let xmax = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ymin = ys.iter().copied().fold(f64::INFINITY, f64::min);
    let ymax = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let xpad = 0.04 * (xmax - xmin).max(1e-12);
    let ypad = 0.08 * (ymax - ymin).max(1e-12);
    let xmin = xmin - xpad;
    let xmax = xmax + xpad;
    let ymin = ymin - ypad;
    let ymax = ymax + ypad;

    let mapx = |x: f64| ml + (tx(x, spec.x_scale) - xmin) / (xmax - xmin) * pw;
    let mapy = |y: f64| mt + (ymax - tx(y, spec.y_scale)) / (ymax - ymin) * ph;

    let colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ];

    let mut s = String::new();
    s.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<rect width="100%" height="100%" fill="white"/>
<style>
text {{ font-family: Arial, Helvetica, sans-serif; fill: #111; }}
.axis {{ stroke: #111; stroke-width: 1.5; }}
.grid {{ stroke: #ddd; stroke-width: 1; }}
</style>
"#));
    s.push_str(&format!(
        r#"<text x="{}" y="30" font-size="22" text-anchor="middle">{}</text>"#,
        ml + pw / 2.0,
        esc(&spec.title)
    ));

    // Grid and ticks.
    for k in 0..=5 {
        let f = k as f64 / 5.0;
        let xp = ml + f * pw;
        let yp = mt + f * ph;
        s.push_str(&format!(r#"<line class="grid" x1="{xp}" y1="{mt}" x2="{xp}" y2="{}"/>"#, mt + ph));
        s.push_str(&format!(r#"<line class="grid" x1="{ml}" y1="{yp}" x2="{}" y2="{yp}"/>"#, ml + pw));

        let xv = xmin + f * (xmax - xmin);
        let yv = ymax - f * (ymax - ymin);
        let xlab = match spec.x_scale {
            AxisScale::Linear => format!("{:.3}", xv),
            AxisScale::Log10 => format!("{:.2e}", 10f64.powf(xv)),
        };
        let ylab = match spec.y_scale {
            AxisScale::Linear => format!("{:.3}", yv),
            AxisScale::Log10 => format!("{:.2e}", 10f64.powf(yv)),
        };
        s.push_str(&format!(r#"<text x="{xp}" y="{}" font-size="12" text-anchor="middle">{}</text>"#, mt + ph + 22.0, esc(&xlab)));
        s.push_str(&format!(r#"<text x="{}" y="{}" font-size="12" text-anchor="end" dominant-baseline="middle">{}</text>"#, ml - 8.0, yp, esc(&ylab)));
    }
    s.push_str(&format!(r#"<line class="axis" x1="{ml}" y1="{}" x2="{}" y2="{}"/>"#, mt + ph, ml + pw, mt + ph));
    s.push_str(&format!(r#"<line class="axis" x1="{ml}" y1="{mt}" x2="{ml}" y2="{}"/>"#, mt + ph));
    s.push_str(&format!(r#"<text x="{}" y="{}" font-size="16" text-anchor="middle">{}</text>"#, ml + pw / 2.0, h - 25.0, esc(&spec.x_label)));
    s.push_str(&format!(r#"<text transform="translate(24,{}) rotate(-90)" font-size="16" text-anchor="middle">{}</text>"#, mt + ph / 2.0, esc(&spec.y_label)));

    for (idx, c) in spec.curves.iter().enumerate() {
        let color = colors[idx % colors.len()];
        if let Some(sd) = &c.sd {
            let mut pts = Vec::<String>::new();
            for ((&x, &y), &d) in c.x.iter().zip(c.y.iter()).zip(sd.iter()) {
                pts.push(format!("{:.2},{:.2}", mapx(x), mapy((y + d).max(1e-300))));
            }
            for ((&x, &y), &d) in c.x.iter().zip(c.y.iter()).zip(sd.iter()).rev() {
                pts.push(format!("{:.2},{:.2}", mapx(x), mapy((y - d).max(1e-300))));
            }
            s.push_str(&format!(r#"<polygon points="{}" fill="{}" fill-opacity="0.14" stroke="none"/>"#, pts.join(" "), color));
        }

        let mut d = String::new();
        for (k, (&x, &y)) in c.x.iter().zip(c.y.iter()).enumerate() {
            let cmd = if k == 0 { "M" } else { "L" };
            d.push_str(&format!("{cmd}{:.2},{:.2} ", mapx(x), mapy(y.max(1e-300))));
        }
        s.push_str(&format!(r#"<path d="{}" fill="none" stroke="{}" stroke-width="2.2"/>"#, d, color));
        for (&x, &y) in c.x.iter().zip(c.y.iter()) {
            s.push_str(&format!(r#"<circle cx="{:.2}" cy="{:.2}" r="3.2" fill="{}"/>"#, mapx(x), mapy(y.max(1e-300)), color));
        }

        let ly = mt + 25.0 + idx as f64 * 22.0;
        let lx = ml + pw + 25.0;
        s.push_str(&format!(r#"<line x1="{lx}" y1="{ly}" x2="{}" y2="{ly}" stroke="{}" stroke-width="2.2"/>"#, lx + 24.0, color));
        s.push_str(&format!(r#"<text x="{}" y="{}" font-size="13" dominant-baseline="middle">{}</text>"#, lx + 30.0, ly, esc(&c.label)));
    }

    s.push_str("</svg>\n");
    fs::write(path, s)?;
    Ok(())
}
