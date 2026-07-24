#!/usr/bin/env python3
"""
Builds the monthly HTML report from collected BA-zone data — in English and Bosnian.

Usage:
  python src/build_report.py                 # both languages, current month
  python src/build_report.py --month 2026-07
Outputs:
  report/index.html      (English)
  report/bs/index.html   (Bosanski)
"""
import argparse
import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "report"
TZ = "Europe/Sarajevo"

C = {
    "lignite": "#8a6f5c", "hardcoal": "#6e5a4a",
    "hydro_res": "#2f6da8", "hydro_ror": "#5b93c4", "hydro_ps": "#8db6d8",
    "wind": "#3f9b94", "solar": "#d9a441", "other": "#a8a29a",
    "load": "#1a1a1a", "import": "#b0653a", "export": "#4a7c46",
    "ink": "#1a1a1a", "muted": "#6b6b6b", "grid": "#e6e3dc", "bg": "#ffffff",
}
TYPE_MAP = [
    ("Lignite", "lignite"), ("Brown coal", "lignite"), ("Hard coal", "hardcoal"),
    ("Hydro Water Reservoir", "hydro_res"), ("Run-of-river", "hydro_ror"),
    ("Pumped Storage", "hydro_ps"), ("Wind", "wind"), ("Solar", "solar"),
]
BUCKET_ORDER = ["lignite", "hardcoal", "hydro_res", "hydro_ror", "hydro_ps", "wind", "solar", "other"]
NEIGHBOR_CODES = ["HR", "RS", "ME"]

MONTHS = {
    "en": ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"],
    "bs": ["januar", "februar", "mart", "april", "maj", "juni", "juli",
           "august", "septembar", "oktobar", "novembar", "decembar"],
}

S = {
 "en": {
  "html_lang": "en",
  "title": "Bosnia &amp; Herzegovina — Power System Report",
  "page_title": "BiH Power",
  "sub": "BA control area (NOSBiH) · source: ENTSO-E Transparency Platform · generated",
  "lang_link": '<a href="bs/">Bosanski</a>',
  "home_link": '<a href="/">← Home</a>',
  "intro": "Bosnia and Herzegovina runs one of Europe's most distinctive small power systems: roughly two-thirds of generation comes from lignite and a third from hydro, demand peaks in winter, and the grid is interconnected with Croatia, Serbia, and Montenegro. Despite headlines about record import <em>costs</em>, the physical balance tells a different story — measured at the borders, BiH has been a net <strong>exporter</strong> in nearly every month of the past two years. The import-bill spikes reflect expensive winter hours, not a year-round deficit.",
  "buckets": {"lignite": "Lignite", "hardcoal": "Hard coal", "hydro_res": "Hydro reservoir",
              "hydro_ror": "Hydro run-of-river", "hydro_ps": "Hydro pumped", "wind": "Wind",
              "solar": "Solar", "other": "Other"},
  "load": "Load",
  "neighbors": {"HR": "Croatia", "RS": "Serbia", "ME": "Montenegro"},
  "tiles": ["Generation", "Load", "Net position", "Hydro share", "Lignite share"],
  "tile_subs": ["", "", "+ export / − import", "of generation", "of generation"],
  "c1_title": "Daily average generation mix and load — {month}",
  "c1_cap": "Daily average generation by source with system load overlaid. Hydro (blues) vs lignite (brown) is the defining balance of the Bosnian system.",
  "c2_title": "Monthly generation by source, last 24 months",
  "c2_cap": "Monthly energy by source over the last two years — the seasonal hydro swing and its dry-year gaps.",
  "c3_title": "Monthly net position (green = net exporter, rust = net importer) — from {src}",
  "c3_src_flows": "cross-border physical flows",
  "c3_src_calc": "generation − load (flow data unavailable)",
  "c3_cap": "Net cross-border position by month.",
  "c4_title": "Net exchange by partner — {month} (GWh, + = export)",
  "c4_cap": "Where the energy went this month, by border.",
  "c5_title": "Average hourly load profile",
  "c5_cap": "Shape of demand: average day this month vs the same month a year earlier.",
  "c5_xlabel": "hour",
  "cap_title": "Installed capacity (latest reported, MW)",
  "cov_title": "Data coverage",
  "cov_head": ["dataset", "rows", "range", "status"],
  "demo": "⚠ DEMO DATA — synthetic pipeline test. Run collect.py with a real ENTSOE_TOKEN.",
  "footer": 'Data refreshed weekly by GitHub Actions in the <a href="https://github.com/FarukDziho/bih-power-data">bih-power-data</a> repository · source: ENTSO-E Transparency Platform (NOSBiH)',
  "cap_types": {},
 },
 "bs": {
  "html_lang": "bs",
  "title": "Bosna i Hercegovina — Izvještaj o elektroenergetskom sistemu",
  "page_title": "Elektroenergetika BiH",
  "sub": "BA kontrolna oblast (NOSBiH) · izvor: ENTSO-E Transparency Platform · generisano",
  "lang_link": '<a href="../">English</a>',
  "home_link": '<a href="/">← Početna</a>',
  "intro": "Bosna i Hercegovina ima jedan od najinteresantnijih malih elektroenergetskih sistema u Evropi: otprilike dvije trećine proizvodnje daje lignit, a trećinu hidroelektrane; potrošnja je najveća zimi, a mreža je povezana sa Hrvatskom, Srbijom i Crnom Gorom. Uprkos naslovima o rekordnim <em>troškovima</em> uvoza, fizički bilans priča drugu priču — mjereno na granicama, BiH je bila neto <strong>izvoznik</strong> struje u gotovo svakom mjesecu posljednje dvije godine. Skokovi računa za uvoz odraz su skupih zimskih sati, a ne cjelogodišnjeg manjka.",
  "buckets": {"lignite": "Lignit", "hardcoal": "Kameni ugalj", "hydro_res": "Hidro — akumulacija",
              "hydro_ror": "Hidro — protočna", "hydro_ps": "Hidro — pumpna", "wind": "Vjetar",
              "solar": "Solar", "other": "Ostalo"},
  "load": "Potrošnja",
  "neighbors": {"HR": "Hrvatska", "RS": "Srbija", "ME": "Crna Gora"},
  "tiles": ["Proizvodnja", "Potrošnja", "Neto pozicija", "Udio hidro", "Udio lignita"],
  "tile_subs": ["", "", "+ izvoz / − uvoz", "u proizvodnji", "u proizvodnji"],
  "c1_title": "Prosječna dnevna proizvodnja po izvorima i potrošnja — {month}",
  "c1_cap": "Dnevni prosjek proizvodnje po izvorima s potrošnjom sistema. Hidro (plavo) naspram lignita (smeđe) — odnos koji definiše bosanski sistem.",
  "c2_title": "Mjesečna proizvodnja po izvorima, posljednja 24 mjeseca",
  "c2_cap": "Mjesečna energija po izvorima u posljednje dvije godine — sezonski zamah hidroenergije i praznine sušnih godina.",
  "c3_title": "Mjesečna neto pozicija (zeleno = neto izvoznik, smeđe = neto uvoznik) — iz {src}",
  "c3_src_flows": "prekograničnih fizičkih tokova",
  "c3_src_calc": "proizvodnje − potrošnje (podaci o tokovima nedostupni)",
  "c3_cap": "Neto prekogranična pozicija po mjesecima.",
  "c4_title": "Neto razmjena po granici — {month} (GWh, + = izvoz)",
  "c4_cap": "Kuda je energija otišla ovog mjeseca, po granicama.",
  "c5_title": "Prosječni satni profil potrošnje",
  "c5_cap": "Oblik potrošnje: prosječan dan ovog mjeseca naspram istog mjeseca prošle godine.",
  "c5_xlabel": "sat",
  "cap_title": "Instalisani kapaciteti (posljednje prijavljeno, MW)",
  "cov_title": "Pokrivenost podataka",
  "cov_head": ["skup podataka", "redova", "raspon", "status"],
  "demo": "⚠ DEMO PODACI — sintetički test. Pokrenite collect.py sa stvarnim ENTSOE_TOKEN-om.",
  "footer": 'Podaci se automatski osvježavaju svakog ponedjeljka (GitHub Actions) u repozitoriju <a href="https://github.com/FarukDziho/bih-power-data">bih-power-data</a> · izvor: ENTSO-E Transparency Platform (NOSBiH)',
  "cap_types": {"Fossil Brown coal/Lignite": "Lignit (mrki ugalj)", "Fossil Hard coal": "Kameni ugalj",
                "Hydro Water Reservoir": "Hidro — akumulacija", "Hydro Run-of-river and poundage": "Hidro — protočna",
                "Hydro Pumped Storage": "Hidro — pumpno-akumulaciona", "Wind Onshore": "Vjetar (kopneni)",
                "Solar": "Solarna", "Other renewable": "Ostali obnovljivi"},
 },
}

plt.rcParams.update({
    "figure.facecolor": C["bg"], "axes.facecolor": C["bg"], "axes.edgecolor": C["grid"],
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "text.color": C["ink"], "axes.labelcolor": C["muted"],
    "xtick.color": C["muted"], "ytick.color": C["muted"], "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
})


def month_name(period: pd.Period, lang: str) -> str:
    m = MONTHS[lang][period.month - 1]
    return f"{m} {period.year}" if lang == "en" else f"{m} {period.year}."


def read(name):
    f = DATA / f"{name}.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(TZ)
    return df


def bucketize(gen: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in gen.columns if "Consumption" not in c]
    out = {}
    for c in cols:
        bucket = "other"
        for kw, b in TYPE_MAP:
            if kw.lower() in c.lower():
                bucket = b
                break
        out.setdefault(bucket, []).append(c)
    agg = pd.DataFrame({b: gen[cs].sum(axis=1) for b, cs in out.items()})
    return agg[[b for b in BUCKET_ORDER if b in agg.columns]]


def fig64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def gwh(series_mw: pd.Series) -> float:
    s = series_mw.resample("h").mean()
    return float(s.sum()) / 1000.0


def build(lang: str, month: pd.Period, state: dict, data: dict) -> str:
    T = S[lang]
    mn = month_name(month, lang)
    m_start = month.to_timestamp().tz_localize(TZ)
    m_end = (month + 1).to_timestamp().tz_localize(TZ)
    prev_year_month = month - 12
    demo = state.get("mode") == "demo"

    load_s, buckets, cap = data["load_s"], data["buckets"], data["cap"]
    flows_in, flows_out = data["flows_in"], data["flows_out"]
    BUCKET_LABEL = T["buckets"]

    figs = {}

    bm = buckets.loc[m_start:m_end]
    lm = load_s.loc[m_start:m_end]
    if len(bm):
        daily = bm.resample("D").mean()
        dload = lm.resample("D").mean()
        fig, ax = plt.subplots(figsize=(9.5, 3.6))
        ax.stackplot(daily.index, [daily[b] for b in daily.columns],
                     colors=[C[b] for b in daily.columns],
                     labels=[BUCKET_LABEL[b] for b in daily.columns], linewidth=0)
        ax.plot(dload.index, dload.values, color=C["load"], lw=2, label=T["load"])
        ax.set_title(T["c1_title"].format(month=mn))
        ax.set_ylabel("MW")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
        ax.margins(x=0.01)
        figs["month_mix"] = fig64(fig)

    b24 = buckets.loc[buckets.index >= m_end - pd.DateOffset(months=24)]
    mgen = b24.resample("MS").mean() * 730 / 1000
    mgen = mgen[mgen.index < m_end]
    if len(mgen):
        fig, ax = plt.subplots(figsize=(9.5, 3.6))
        bottom = None
        for b in mgen.columns:
            ax.bar(mgen.index, mgen[b], width=20, bottom=bottom, color=C[b],
                   label=BUCKET_LABEL[b], linewidth=0)
            bottom = mgen[b] if bottom is None else bottom + mgen[b]
        ax.set_title(T["c2_title"])
        ax.set_ylabel("GWh")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.margins(x=0.01)
        figs["mix_24m"] = fig64(fig)

    if flows_in and flows_out:
        imp = pd.concat(flows_in.values(), axis=1).sum(axis=1)
        exp = pd.concat(flows_out.values(), axis=1).sum(axis=1)
        net = (exp - imp).resample("MS").mean() * 730 / 1000
        src_note = T["c3_src_flows"]
    else:
        net = (buckets.sum(axis=1) - load_s).resample("MS").mean() * 730 / 1000
        src_note = T["c3_src_calc"]
    net = net[(net.index >= m_end - pd.DateOffset(months=24)) & (net.index < m_end)]
    if len(net):
        fig, ax = plt.subplots(figsize=(9.5, 3.2))
        colors = [C["export"] if v >= 0 else C["import"] for v in net.values]
        ax.bar(net.index, net.values, width=20, color=colors, linewidth=0)
        ax.axhline(0, color=C["muted"], lw=0.8)
        ax.set_title(T["c3_title"].format(src=src_note))
        ax.set_ylabel("GWh")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.margins(x=0.01)
        figs["net_pos"] = fig64(fig)

    if flows_in and flows_out:
        rows = []
        for n in NEIGHBOR_CODES:
            i = gwh(flows_in[n].loc[m_start:m_end]) if n in flows_in and len(flows_in[n].loc[m_start:m_end]) else 0.0
            o = gwh(flows_out[n].loc[m_start:m_end]) if n in flows_out and len(flows_out[n].loc[m_start:m_end]) else 0.0
            rows.append((T["neighbors"][n], o - i))
        if rows:
            fig, ax = plt.subplots(figsize=(6.4, 2.8))
            labels = [r[0] for r in rows]
            vals = [r[1] for r in rows]
            colors = [C["export"] if v >= 0 else C["import"] for v in vals]
            ax.barh(labels, vals, color=colors, height=0.55, linewidth=0)
            ax.axvline(0, color=C["muted"], lw=0.8)
            for lbl, v in zip(labels, vals):
                ax.text(v + (2 if v >= 0 else -2), lbl, f"{v:+.0f}", va="center",
                        ha="left" if v >= 0 else "right", fontsize=9, color=C["ink"])
            ax.set_title(T["c4_title"].format(month=mn))
            figs["partners"] = fig64(fig)

    prof_now = lm.groupby(lm.index.hour).mean() if len(lm) else None
    lp = load_s.loc[prev_year_month.to_timestamp().tz_localize(TZ):(prev_year_month + 1).to_timestamp().tz_localize(TZ)]
    prof_prev = lp.groupby(lp.index.hour).mean() if len(lp) else None
    if prof_now is not None and len(prof_now):
        fig, ax = plt.subplots(figsize=(6.4, 2.9))
        ax.plot(prof_now.index, prof_now.values, color=C["load"], lw=2, label=month_name(month, lang))
        if prof_prev is not None and len(prof_prev):
            ax.plot(prof_prev.index, prof_prev.values, color=C["muted"], lw=1.6, ls="--",
                    label=month_name(prev_year_month, lang))
        ax.set_title(T["c5_title"])
        ax.set_ylabel("MW"); ax.set_xlabel(T["c5_xlabel"])
        ax.set_xticks(range(0, 24, 3))
        ax.legend(frameon=False, fontsize=8.5)
        figs["profile"] = fig64(fig)

    tot_gen = gwh(bm.sum(axis=1)) if len(bm) else float("nan")
    tot_load = gwh(lm) if len(lm) else float("nan")
    hydro_cols = [b for b in ("hydro_res", "hydro_ror", "hydro_ps") if b in buckets.columns]
    hydro_share = (gwh(bm[hydro_cols].sum(axis=1)) / tot_gen * 100) if len(bm) and tot_gen else float("nan")
    lig_share = (gwh(bm["lignite"]) / tot_gen * 100) if len(bm) and "lignite" in bm and tot_gen else float("nan")
    if flows_in and flows_out:
        net_m = sum(gwh(flows_out[n].loc[m_start:m_end]) for n in flows_out) - \
                sum(gwh(flows_in[n].loc[m_start:m_end]) for n in flows_in)
    else:
        net_m = tot_gen - tot_load

    vals = [f"{tot_gen:,.0f} GWh", f"{tot_load:,.0f} GWh", f"{net_m:+,.0f} GWh",
            f"{hydro_share:,.0f}%", f"{lig_share:,.0f}%"]
    subs = [mn, mn] + T["tile_subs"][2:]
    tiles = "".join(
        f'<div class="tile"><div class="v">{v}</div><div class="l">{l}</div><div class="s">{s}</div></div>'
        for v, l, s in zip(vals, T["tiles"], subs))

    cap_html = ""
    if cap is not None and len(cap):
        last = cap.iloc[-1].dropna().sort_values(ascending=False)
        rows = "".join(
            f"<tr><td>{T['cap_types'].get(k, k)}</td><td class='num'>{v:,.0f}</td></tr>"
            for k, v in last.items())
        cap_html = f"<h2>{T['cap_title']}</h2><table><tbody>{rows}</tbody></table>"

    cov_rows = ""
    for name, meta in sorted(state.get("datasets", {}).items()):
        rng = f"{meta.get('start','')[:10]} → {meta.get('end','')[:10]}" if meta.get("rows") else "—"
        cov_rows += (f"<tr><td><code>{name}</code></td><td class='num'>{meta.get('rows',0):,}</td>"
                     f"<td>{rng}</td><td>{meta.get('status','?')}</td></tr>")
    hd = T["cov_head"]
    cov_html = (f"<h2>{T['cov_title']}</h2><table><thead><tr><th>{hd[0]}</th><th>{hd[1]}</th>"
                f"<th>{hd[2]}</th><th>{hd[3]}</th></tr></thead><tbody>{cov_rows}</tbody></table>")

    demo_banner = f'<div class="demo">{T["demo"]}</div>' if demo else ""

    def img(key, caption=""):
        if key not in figs:
            return ""
        return (f'<figure><img src="data:image/png;base64,{figs[key]}" alt="{caption}">'
                f'<figcaption>{caption}</figcaption></figure>')

    return f"""<!doctype html><html lang="{T['html_lang']}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{T['page_title']} — {mn}</title>
<style>
 body{{font-family:Iowan Old Style,Charter,Georgia,serif;background:#fbfaf7;color:#1a1a1a;margin:0;line-height:1.55}}
 .wrap{{max-width:62rem;margin:0 auto;padding:2.2rem 1.2rem 4rem}}
 h1{{font-size:1.7rem;margin:.2rem 0 .2rem}} h2{{font-size:1.15rem;margin:2rem 0 .6rem}}
 .sub{{color:#6b6b6b;font-size:.95rem}}
 .langsw{{float:right;font-family:-apple-system,Segoe UI,Helvetica,sans-serif;font-size:.85rem}}
 .homelink{{font-family:-apple-system,Segoe UI,Helvetica,sans-serif;font-size:.85rem;margin-bottom:.6rem}}
 .homelink a,.langsw a{{color:#0b4f86;text-decoration:none}}
 .homelink a:hover,.langsw a:hover{{text-decoration:underline}}
 .intro{{max-width:46rem;font-size:1rem;margin:.9rem 0 0}}
 .tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.7rem;margin:1.4rem 0}}
 .tile{{border:1px solid #e6e3dc;border-radius:12px;background:#fff;padding:.8rem 1rem}}
 .tile .v{{font-size:1.35rem;font-weight:700;font-family:-apple-system,Segoe UI,Helvetica,sans-serif}}
 .tile .l{{font-size:.8rem;color:#6b6b6b;font-family:-apple-system,Segoe UI,Helvetica,sans-serif}}
 .tile .s{{font-size:.72rem;color:#9b968c;font-family:-apple-system,Segoe UI,Helvetica,sans-serif}}
 figure{{margin:1.2rem 0;border:1px solid #e6e3dc;border-radius:12px;background:#fff;padding:1rem}}
 figure img{{width:100%;height:auto;display:block}}
 figcaption{{font-size:.78rem;color:#6b6b6b;padding-top:.5rem;font-family:-apple-system,Segoe UI,Helvetica,sans-serif}}
 table{{border-collapse:collapse;background:#fff;border:1px solid #e6e3dc;border-radius:10px;font-size:.86rem;min-width:20rem}}
 th,td{{padding:.35rem .8rem;border-bottom:1px solid #f0ede6;text-align:left;font-family:-apple-system,Segoe UI,Helvetica,sans-serif}}
 td.num{{text-align:right;font-variant-numeric:tabular-nums}}
 .demo{{background:#fff3cd;border:1px solid #e0c060;border-radius:10px;padding:.6rem 1rem;margin:1rem 0;font-family:-apple-system,Segoe UI,Helvetica,sans-serif;font-size:.85rem}}
 footer{{margin-top:3rem;color:#6b6b6b;font-size:.78rem;font-family:-apple-system,Segoe UI,Helvetica,sans-serif}}
 code{{font-size:.8rem}}
</style></head><body><div class="wrap">
<div class="langsw">{T['lang_link']}</div>
<div class="homelink">{T['home_link']}</div>
<h1>{T['title']}</h1>
<div class="sub">{mn} · {T['sub']} {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</div>
<p class="intro">{T['intro']}</p>
{demo_banner}
<div class="tiles">{tiles}</div>
{img('month_mix', T['c1_cap'])}
{img('mix_24m', T['c2_cap'])}
{img('net_pos', T['c3_cap'])}
{img('partners', T['c4_cap'])}
{img('profile', T['c5_cap'])}
{cap_html}
{cov_html}
<footer>{T['footer']}</footer>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default=None, help="YYYY-MM (default: current month)")
    args = ap.parse_args()

    now = pd.Timestamp.now(tz=TZ)
    month = pd.Period(args.month, "M") if args.month else now.to_period("M")

    state = json.loads((DATA / "state.json").read_text()) if (DATA / "state.json").exists() else {}
    load = read("load_actual")
    gen = read("generation_per_type")
    if load is None or gen is None:
        raise SystemExit("Missing core datasets — run collect.py first.")

    data = {
        "load_s": load.iloc[:, 0],
        "buckets": bucketize(gen),
        "cap": read("installed_capacity"),
        "flows_in": {}, "flows_out": {},
    }
    for n in NEIGHBOR_CODES:
        fi, fo = read(f"flow_{n}_BA"), read(f"flow_BA_{n}")
        if fi is not None: data["flows_in"][n] = fi.iloc[:, 0]
        if fo is not None: data["flows_out"][n] = fo.iloc[:, 0]

    OUT.mkdir(exist_ok=True)
    (OUT / "bs").mkdir(exist_ok=True)
    (OUT / "index.html").write_text(build("en", month, state, data), encoding="utf-8")
    (OUT / "bs" / "index.html").write_text(build("bs", month, state, data), encoding="utf-8")
    print(f"report/index.html + report/bs/index.html written ({month})")


if __name__ == "__main__":
    main()
