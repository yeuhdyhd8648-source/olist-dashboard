# -*- coding: utf-8 -*-
"""
Olist Executive Dashboard  |  لوحة القيادة التنفيذية لبيانات Olist
=====================================================================
Streamlit + Plotly  ·  ملف واحد جاهز للنشر على Streamlit Community Cloud.

طريقة التشغيل المحلي:
    pip install -r requirements.txt
    streamlit run app.py

ضع ملف البيانات Olist_Master_Analysis.csv بجانب app.py (أو داخل مجلد data/).
إن لم يوجد الملف يمكن رفعه من الشريط الجانبي.
"""
from __future__ import annotations

import inspect
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Olist | لوحة القيادة التنفيذية",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
# 1) الثوابت والهوية البصرية
# ══════════════════════════════════════════════════════════════════════
DATA_FILENAME = "Olist_Master_Analysis.csv"

# الملف الأصلي بلا صف عناوين — الترتيب مستنتج من البيانات
RAW_COLUMNS = [
    "order_id", "status", "purchase_ts", "delivered_ts", "cust_state", "cust_city",
    "segment_raw", "qty", "order_total", "category", "price", "freight", "item_total",
    "seller_id", "seller_city", "seller_state", "pay_type", "installments",
    "pay_value", "review", "review_label",
]
NUMERIC_COLS = ["qty", "order_total", "price", "freight", "item_total",
                "installments", "pay_value", "review"]

C = dict(
    navy="#0B2545", blue="#2563EB", sky="#60A5FA", teal="#14B8A6", amber="#F59E0B",
    red="#E11D48", slate="#64748B", border="#E2E8F0", text="#0F172A", grid="#EDF2F7",
)
PALETTE = [C["blue"], C["teal"], C["amber"], C["navy"], C["red"], C["slate"]]
FONT = "Tajawal, Segoe UI, Tahoma, sans-serif"

# شرائح قيمة الطلب الواحد (Order Value Segmentation) — بديل تسمية VIP/Gold
SEG_TOP = "طلبات كبرى (+1,000)"
SEG_HIGH = "طلبات مرتفعة (500–1,000)"
SEG_STD = "طلبات قياسية (<500)"
SEG_ORDER = [SEG_TOP, SEG_HIGH, SEG_STD]
SEG_COLORS = {SEG_TOP: C["navy"], SEG_HIGH: C["blue"], SEG_STD: C["teal"]}

BUCKET_LABELS = ["أقل من 7 أيام", "7–14 يوماً", "14–21 يوماً", "أكثر من 21 يوماً"]
SCORE_COLORS = {1: C["red"], 2: "#F97316", 3: C["amber"], 4: "#34D399", 5: C["teal"]}

PAY_LABELS = {"credit_card": "بطاقة ائتمان", "boleto": "بوليتو (تحويل)",
              "voucher": "قسيمة", "debit_card": "بطاقة خصم"}

REGION_MAP = {
    **dict.fromkeys(["AC", "AP", "AM", "PA", "RO", "RR", "TO"], "الشمال"),
    **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "الشمال الشرقي"),
    **dict.fromkeys(["DF", "GO", "MT", "MS"], "الوسط الغربي"),
    **dict.fromkeys(["ES", "MG", "RJ", "SP"], "الجنوب الشرقي"),
    **dict.fromkeys(["PR", "RS", "SC"], "الجنوب"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');
.stApp{background:#F4F7FB;font-family:'Tajawal','Segoe UI',Tahoma,sans-serif;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{direction:rtl;padding-top:1.2rem;padding-bottom:3rem;max-width:1440px;}
.block-container p,.block-container li,.block-container label{text-align:right;}
[data-testid="stSidebar"]{background:#FFFFFF;border-left:1px solid #E2E8F0;}
[data-testid="stSidebarUserContent"]{direction:rtl;text-align:right;}
.js-plotly-plot,.plotly,.plot-container{direction:ltr;}
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:14px;background:#FFFFFF;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{direction:rtl;gap:6px;border-bottom:1px solid #E2E8F0;}
.stTabs [data-baseweb="tab"]{height:46px;padding:0 18px;border-radius:10px 10px 0 0;font-weight:700;color:#64748B;}
.stTabs [aria-selected="true"]{color:#0B2545!important;background:#FFFFFF;}
.stTabs [data-baseweb="tab-highlight"]{background:#2563EB!important;height:3px;}

/* Hero */
.hero{background:linear-gradient(120deg,#0B2545 0%,#13407A 55%,#2563EB 100%);border-radius:18px;
padding:26px 30px;color:#fff;margin-bottom:18px;box-shadow:0 8px 24px rgba(11,37,69,.18);}
.hero-title{color:#fff;font-size:1.85rem;font-weight:800;margin:0 0 6px 0;}
.hero-sub{color:#CFE0FF;font-size:1rem;margin:0;}
.chip{display:inline-block;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.28);
border-radius:999px;padding:4px 13px;margin:14px 0 0 8px;font-size:.82rem;color:#fff;}

/* KPI cards */
.kpi{background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:15px 18px 14px;
box-shadow:0 1px 3px rgba(15,23,42,.05);height:100%;margin-bottom:10px;}
.kpi-label{color:#64748B;font-size:.86rem;font-weight:700;text-align:right;}
.kpi-value{color:#0B2545;font-size:1.85rem;font-weight:800;line-height:1.25;direction:ltr;text-align:right;margin:2px 0;}
.kpi-sub{color:#94A3B8;font-size:.78rem;text-align:right;}

/* Titles */
.sec{display:flex;align-items:center;gap:10px;margin:22px 0 10px 0;}
.sec-bar{width:5px;height:24px;background:#2563EB;border-radius:4px;}
.sec-t{font-size:1.2rem;font-weight:800;color:#0B2545;}
.sec-s{font-size:.85rem;color:#64748B;margin-right:6px;}
.ct{font-size:1.02rem;font-weight:800;color:#0B2545;text-align:right;}
.cs{font-size:.8rem;color:#64748B;text-align:right;margin-bottom:4px;}
.insight{background:#EFF6FF;border-right:4px solid #2563EB;border-radius:10px;padding:10px 14px;
color:#1E3A8A;font-size:.9rem;text-align:right;margin-top:6px;}
.insight.warn{background:#FFF1F2;border-right-color:#E11D48;color:#9F1239;}

/* Tables */
.tbl-wrap{overflow-x:auto;border:1px solid #E2E8F0;border-radius:12px;background:#fff;}
.tbl{width:100%;border-collapse:collapse;font-size:.88rem;}
.tbl th{background:#0B2545;color:#fff;padding:10px 12px;text-align:right!important;font-weight:700;white-space:nowrap;}
.tbl td{padding:9px 12px;border-bottom:1px solid #EEF2F7;text-align:right;white-space:nowrap;color:#0F172A;}
.tbl tr:nth-child(even) td{background:#F8FAFC;}
.bdg{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:700;font-size:.8rem;}
.bdg-g{background:#CCFBF1;color:#0F766E;}.bdg-a{background:#FEF3C7;color:#92400E;}.bdg-r{background:#FFE4E6;color:#BE123C;}

/* Story */
.story-hero{background:linear-gradient(120deg,#0B2545,#13407A);color:#fff;border-radius:16px;padding:24px 28px;margin-bottom:16px;}
.story-hero .h{font-size:1.45rem;font-weight:800;margin-bottom:8px;text-align:right;}
.story-hero .p{color:#D6E4FF;font-size:1rem;line-height:1.9;text-align:right;}
.finding{background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:18px 20px;height:100%;
box-shadow:0 1px 3px rgba(15,23,42,.05);}
.finding .n{display:inline-block;background:#0B2545;color:#fff;border-radius:8px;padding:1px 10px;font-weight:800;margin-bottom:8px;}
.finding .t{font-size:1.05rem;font-weight:800;color:#0B2545;margin-bottom:6px;text-align:right;}
.finding .b{color:#334155;font-size:.92rem;line-height:1.85;text-align:right;}
.big{font-size:1.7rem;font-weight:800;direction:ltr;display:inline-block;}
.red{color:#E11D48;}.teal{color:#0F9E90;}.navy{color:#0B2545;}
.rec{background:#fff;border:1px solid #E2E8F0;border-right:5px solid #2563EB;border-radius:12px;
padding:15px 20px;margin-bottom:12px;box-shadow:0 1px 3px rgba(15,23,42,.04);}
.rec.hi{border-right-color:#E11D48;}.rec.md{border-right-color:#F59E0B;}
.rec .t{font-weight:800;color:#0B2545;font-size:1.02rem;text-align:right;}
.rec .b{color:#334155;font-size:.92rem;line-height:1.85;text-align:right;margin-top:4px;}
.rec .k{color:#0F766E;font-weight:700;font-size:.88rem;margin-top:6px;text-align:right;}
.pill{display:inline-block;padding:1px 10px;border-radius:999px;font-size:.75rem;font-weight:800;margin-left:8px;}
.pill.hi{background:#FFE4E6;color:#BE123C;}.pill.md{background:#FEF3C7;color:#92400E;}
.foot{color:#94A3B8;font-size:.8rem;text-align:center;margin-top:28px;}
</style>
"""


# ══════════════════════════════════════════════════════════════════════
# 2) أدوات مساعدة (تنسيق وعرض)
# ══════════════════════════════════════════════════════════════════════
def _compact(html: str) -> str:
    """يضغط الـ HTML في سطر واحد لتفادي تحويل Markdown له إلى code-block."""
    joined = " ".join(line.strip() for line in html.splitlines() if line.strip())
    return joined.replace("$", "&#36;")  # منع تفسير $ كمعادلات LaTeX في markdown


def md(html: str) -> None:
    st.markdown(_compact(html), unsafe_allow_html=True)


def fmt_int(x) -> str:
    return "—" if pd.isna(x) else f"{x:,.0f}"


def fmt_money(x) -> str:
    if pd.isna(x):
        return "—"
    if abs(x) >= 1e6:
        return f"R$ {x / 1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"R$ {x / 1e3:,.1f}K"
    return f"R$ {x:,.0f}"


def fmt_pct(x, d: int = 1) -> str:
    return "—" if pd.isna(x) else f"{x * 100:.{d}f}%"


def fmt_num(x, d: int = 1) -> str:
    return "—" if pd.isna(x) else f"{x:,.{d}f}"


def pretty_cat(s: str) -> str:
    return str(s).replace("_", " ")


_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def show(fig: go.Figure, key: str) -> None:
    """st.plotly_chart متوافق مع الإصدارات القديمة والجديدة من Streamlit."""
    cfg = {"displayModeBar": False}
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch", config=cfg, key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, config=cfg, key=key)


def style(fig: go.Figure, height: int = 370) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=8, r=8, t=28, b=8),
        font=dict(family=FONT, size=13, color=C["text"]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=PALETTE, hoverlabel=dict(font=dict(family=FONT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
    )
    fig.update_xaxes(showgrid=False, linecolor=C["border"])
    fig.update_yaxes(gridcolor=C["grid"], zeroline=False)
    return fig


def section(title: str, sub: str = "") -> None:
    md(f'<div class="sec"><div class="sec-bar"></div><div class="sec-t">{title}</div>'
       f'<div class="sec-s">{sub}</div></div>')


def chart_header(title: str, sub: str = "") -> None:
    md(f'<div class="ct">{title}</div><div class="cs">{sub}</div>')


def insight(text: str, warn: bool = False) -> None:
    md(f'<div class="insight{" warn" if warn else ""}">{text}</div>')


def kpi_card(label: str, value: str, sub: str = "", accent: str = C["blue"]) -> str:
    return _compact(
        f'<div class="kpi" style="border-top:3px solid {accent}">'
        f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>'
    )


def html_table(df: pd.DataFrame) -> str:
    body = df.to_html(index=False, escape=False, border=0, classes="tbl", na_rep="—")
    return _compact(f'<div class="tbl-wrap">{body}</div>')


def rating_badge(r: float) -> str:
    if pd.isna(r):
        return "—"
    cls = "bdg-r" if r < 3.8 else ("bdg-a" if r < 4.0 else "bdg-g")
    return f'<span class="bdg {cls}">{r:.2f}</span>'


# ══════════════════════════════════════════════════════════════════════
# 3) الهندسة البرمجية للبيانات
# ══════════════════════════════════════════════════════════════════════
def map_segment(label: str) -> str:
    """تحويل تصنيف VIP/Gold (المبني على قيمة الطلب) إلى تسمية تعبّر عن ذلك."""
    s = str(label)
    if "VIP" in s:
        return SEG_TOP
    if "Gold" in s:
        return SEG_HIGH
    return SEG_STD


def prepare(raw: pd.DataFrame):
    """
    يبني DataFrame البنود (order_items) وDataFrame الطلبات الفريدة (unique_orders).
    - order_items : كل صف = بند ← إيراد البند = price + freight
    - unique_orders : drop_duplicates على order_id ← لا تضخم في المدفوعات
    """
    raw = raw.copy()
    # في حال أضاف المستخدم صف عناوين لاحقاً
    if str(raw.iloc[0, 0]).strip().lower() == "order_id":
        raw = raw.iloc[1:].reset_index(drop=True)
    for c in NUMERIC_COLS:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    dq = {
        "rows": len(raw),
        "orders": int(raw["order_id"].nunique()),
        "sellers": int(raw["seller_id"].nunique()),
        "dup_rows": int(raw.duplicated().sum()),
        "missing_category": int(raw["category"].isna().sum()),
        "missing_delivery": int(raw["delivered_ts"].isna().sum()),
    }

    # ── التقييمات: "بدون تقييم" ← NaN + Flag ──────────────────────────
    unrated = raw["review_label"].astype(str).str.contains("بدون", na=False)
    raw["is_unreviewed_row"] = unrated
    raw.loc[unrated | ~raw["review"].between(1, 5), "review"] = np.nan
    dq["unreviewed_rows"] = int(unrated.sum())

    # ── التواريخ وزمن التسليم ─────────────────────────────────────────
    raw["purchase_ts"] = pd.to_datetime(raw["purchase_ts"].astype(str).str[:19], errors="coerce")
    raw["delivered_ts"] = pd.to_datetime(raw["delivered_ts"].astype(str).str[:19], errors="coerce")
    raw["purchase_month"] = raw["purchase_ts"].dt.to_period("M").dt.to_timestamp()
    raw["delivery_days"] = (raw["delivered_ts"] - raw["purchase_ts"]).dt.total_seconds() / 86400
    raw["delivery_bucket"] = pd.cut(raw["delivery_days"], bins=[-np.inf, 7, 14, 21, np.inf],
                                    labels=BUCKET_LABELS, right=False)

    # ── الشرائح والجغرافيا والفئات ────────────────────────────────────
    raw["segment"] = raw["segment_raw"].map(map_segment)
    raw["category"] = raw["category"].fillna("غير مصنّف").map(pretty_cat)
    raw["cust_region"] = raw["cust_state"].map(REGION_MAP).fillna("غير معروف")
    raw["is_same_state"] = raw["cust_state"] == raw["seller_state"]

    # ── إيراد البند بدقة + Flag التقييم السلبي ─────────────────────────
    raw["item_revenue"] = (raw["price"] + raw["freight"]).round(2)
    raw["is_bad"] = np.where(raw["review"].isna(), np.nan, (raw["review"] <= 2).astype(float))

    # ═══ DataFrame البنود ═══
    item_cols = ["order_id", "purchase_ts", "purchase_month", "cust_state", "cust_region", "segment",
                 "category", "price", "freight", "item_revenue", "seller_id", "seller_city",
                 "seller_state", "pay_type", "review", "is_bad", "delivery_days", "is_same_state"]
    order_items = raw[item_cols].reset_index(drop=True)

    # ═══ DataFrame الطلبات الفريدة ═══
    first_cols = ["order_id", "purchase_ts", "purchase_month", "cust_state", "cust_city", "cust_region",
                  "segment", "order_total", "pay_type", "installments", "seller_state",
                  "is_same_state", "delivery_days", "delivery_bucket"]
    unique_orders = raw.drop_duplicates(subset="order_id", keep="first")[first_cols].copy()
    agg = raw.groupby("order_id", sort=False).agg(
        items_count=("item_revenue", "size"), items_gmv=("item_revenue", "sum"),
        price_total=("price", "sum"), freight_total=("freight", "sum"), review=("review", "mean"),
    )
    unique_orders = unique_orders.merge(agg, left_on="order_id", right_index=True, how="left")
    unique_orders["is_unreviewed"] = unique_orders["review"].isna()
    unique_orders["is_bad"] = np.where(unique_orders["review"].isna(), np.nan,
                                       (unique_orders["review"] <= 2).astype(float))
    unique_orders["freight_ratio"] = (unique_orders["freight_total"] /
                                      unique_orders["price_total"].replace(0, np.nan))
    unique_orders = unique_orders.reset_index(drop=True)
    dq["unreviewed_orders"] = int(unique_orders["is_unreviewed"].sum())
    return order_items, unique_orders, dq


@st.cache_data(show_spinner="جارٍ تحميل البيانات ومعالجتها …")
def load_and_prepare(source):
    buf = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    raw = pd.read_csv(buf, header=None, names=RAW_COLUMNS, encoding="utf-8-sig", low_memory=False)
    return prepare(raw)


def find_data_file():
    here = Path(__file__).resolve().parent
    for p in (here / DATA_FILENAME, here / "data" / DATA_FILENAME,
              Path.cwd() / DATA_FILENAME, Path.cwd() / "data" / DATA_FILENAME):
        if p.exists():
            return str(p)
    return None


# ══════════════════════════════════════════════════════════════════════
# 4) طبقة التحليل (KPIs، شرائح، بائعون، القصة)
# ══════════════════════════════════════════════════════════════════════
def compute_kpis(items: pd.DataFrame, orders: pd.DataFrame) -> dict:
    n = len(orders)
    gmv = items["item_revenue"].sum()
    d = orders["delivery_days"]
    return dict(
        n_orders=n, n_items=len(items), gmv=gmv, payments=orders["order_total"].sum(),
        aov=orders["order_total"].mean(), items_per_order=len(items) / n if n else np.nan,
        freight_share=items["freight"].sum() / gmv if gmv else np.nan,
        freight_per_item=items["freight"].mean(),
        avg_review=orders["review"].mean(), bad_rate=orders["is_bad"].mean(),
        unreviewed_rate=orders["is_unreviewed"].mean(), unreviewed_n=int(orders["is_unreviewed"].sum()),
        avg_days=d.mean(), med_days=d.median(), p90_days=d.quantile(0.9),
        same_state=orders["is_same_state"].mean(), sellers=items["seller_id"].nunique(),
    )


def segment_summary(items: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    o = orders.groupby("segment").agg(
        orders=("order_id", "size"), payments=("order_total", "sum"), aov=("order_total", "mean"),
        avg_review=("review", "mean"), bad=("is_bad", "mean"), days=("delivery_days", "mean"),
        freight_order=("freight_total", "mean"))
    i = items.groupby("segment").agg(
        gmv=("item_revenue", "sum"), price=("price", "sum"), freight=("freight", "sum"))
    seg = o.join(i)
    seg["pct_orders"] = seg["orders"] / seg["orders"].sum()
    seg["pct_gmv"] = seg["gmv"] / seg["gmv"].sum()
    seg["freight_pct_price"] = seg["freight"] / seg["price"]
    return seg.reindex([s for s in SEG_ORDER if s in seg.index])


def seller_table(items: pd.DataFrame) -> pd.DataFrame:
    g = items.groupby("seller_id").agg(
        gmv=("item_revenue", "sum"), items=("item_revenue", "size"), orders=("order_id", "nunique"),
        avg_review=("review", "mean"), avg_days=("delivery_days", "mean"), bad=("is_bad", "mean"),
        city=("seller_city", "first"), state=("seller_state", "first"))
    g = g.sort_values("gmv", ascending=False)
    g["share"] = g["gmv"] / g["gmv"].sum()
    g["cum_share"] = g["share"].cumsum()
    g = g.reset_index()
    g["short_id"] = g["seller_id"].str[:8]
    return g


def sellers_for_share(g: pd.DataFrame, target: float = 0.8) -> int:
    return int(np.searchsorted(g["cum_share"].values, target) + 1)


def compute_story(items: pd.DataFrame, orders: pd.DataFrame) -> dict:
    """كل أرقام القصة محسوبة من البيانات الكاملة (بغض النظر عن الفلاتر)."""
    S: dict = {}
    k = compute_kpis(items, orders)
    S["k"] = k
    seg = segment_summary(items, orders)
    S["seg"] = seg
    top, high, std = seg.loc[SEG_TOP], seg.loc[SEG_HIGH], seg.loc[SEG_STD]
    S.update(
        bad_top=top["bad"], bad_high=high["bad"], bad_std=std["bad"],
        rev_top=top["avg_review"], rev_std=std["avg_review"],
        days_top=top["days"], days_std=std["days"],
        frt_top=top["freight_order"], frt_std=std["freight_order"],
        prem_orders=seg.loc[[SEG_TOP, SEG_HIGH], "pct_orders"].sum(),
        prem_gmv=seg.loc[[SEG_TOP, SEG_HIGH], "pct_gmv"].sum(),
        top_orders=top["pct_orders"], top_gmv=top["pct_gmv"],
    )
    prem_bad = items[(items["is_bad"] == 1) & items["segment"].isin([SEG_TOP, SEG_HIGH])]
    S["risk_gmv"] = prem_bad["item_revenue"].sum()

    # التسليم والرضا
    d = orders.dropna(subset=["review", "delivery_days"])
    gd = d.groupby(d["review"].round())["delivery_days"].mean()
    S["days_5"], S["days_1"] = gd.get(5.0, np.nan), gd.get(1.0, np.nan)
    S["corr_days"] = d["review"].corr(d["delivery_days"])
    S["corr_freight"] = orders["review"].corr(orders["freight_ratio"])
    bkt = orders.groupby("delivery_bucket", observed=True)["is_bad"].mean()
    S["bad_fast"], S["bad_slow"] = bkt.iloc[0], bkt.iloc[-1]
    S["slow_share"] = (orders["delivery_bucket"] == BUCKET_LABELS[-1]).mean()

    # الجغرافيا
    ss = orders.groupby("is_same_state")["delivery_days"].mean()
    S["days_same"], S["days_cross"] = ss.get(True, np.nan), ss.get(False, np.nan)
    S["cross_share"] = 1 - orders["is_same_state"].mean()
    fs = items.groupby("is_same_state")["freight"].mean()
    S["frt_same"], S["frt_cross"] = fs.get(True, np.nan), fs.get(False, np.nan)
    st_tab = orders.groupby("cust_state").agg(n=("order_id", "size"), days=("delivery_days", "mean"),
                                              review=("review", "mean"))
    base_days = st_tab.loc["SP", "days"] if "SP" in st_tab.index else st_tab["days"].min()
    S["sp_days"] = base_days
    cand = st_tab[(st_tab["n"] >= 0.01 * len(orders)) & (st_tab.index != "SP")].copy()
    cand["excess"] = cand["n"] * (cand["days"] - base_days)
    S["slow_states"] = cand.sort_values("excess", ascending=False).head(3)

    # النمو
    mo = orders.groupby("purchase_month").size()
    full = mo[mo >= 1000]
    S["peak_label"] = full.idxmax().strftime("%Y-%m")
    S["peak_orders"] = int(full.max())
    S["last6_avg"] = float(full.tail(6).mean())
    S["last6_ratio"] = S["last6_avg"] / S["peak_orders"]

    # البائعون
    sel = seller_table(items)
    S["sellers"] = sel
    S["n80"] = sellers_for_share(sel)
    S["n80_pct"] = S["n80"] / len(sel)
    S["top10_share"] = sel.head(10)["share"].sum()
    t10 = sel.head(10).dropna(subset=["avg_review"])
    S["worst_top"] = t10.loc[t10["avg_review"].idxmin()]
    S["worst_top_rank"] = int(sel.index[sel["seller_id"] == S["worst_top"]["seller_id"]][0] + 1)

    # الفئات
    cat = items.groupby("category").agg(gmv=("item_revenue", "sum"), rev=("review", "mean"))
    cat["share"] = cat["gmv"] / cat["gmv"].sum()
    big = cat[(cat["share"] >= 0.015) & (cat.index != "غير مصنّف")]
    S["cat_worst"], S["cat_best"] = big["rev"].idxmin(), big["rev"].idxmax()
    S["cat_worst_row"], S["cat_best_row"] = big.loc[S["cat_worst"]], big.loc[S["cat_best"]]
    return S


# ══════════════════════════════════════════════════════════════════════
# 5) الرسومات البيانية (Plotly)
# ══════════════════════════════════════════════════════════════════════
def fig_monthly(items, orders):
    df = pd.concat([items.groupby("purchase_month")["item_revenue"].sum().rename("gmv"),
                    orders.groupby("purchase_month").size().rename("orders")], axis=1).sort_index()
    big = df[df["orders"] >= 30]
    df = big if len(big) >= 3 else df
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df.index, y=df["gmv"], name="GMV (R$)", marker_color=C["blue"],
                         hovertemplate="%{x|%Y-%m}<br>GMV: R$ %{y:,.0f}<extra></extra>"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df["orders"], name="عدد الطلبات", mode="lines+markers",
                             line=dict(color=C["amber"], width=3), marker=dict(size=7),
                             hovertemplate="%{x|%Y-%m}<br>الطلبات: %{y:,.0f}<extra></extra>"),
                  secondary_y=True)
    style(fig, 380)
    fig.update_yaxes(tickformat=",.2s", secondary_y=False)
    fig.update_yaxes(showgrid=False, tickformat=",.0f", secondary_y=True)
    return fig


def fig_segment_share(seg):
    fig = go.Figure()
    fig.add_trace(go.Bar(name="حصة الطلبات", x=seg.index, y=seg["pct_orders"] * 100,
                         marker_color=C["sky"], text=[fmt_pct(v) for v in seg["pct_orders"]],
                         textposition="outside"))
    fig.add_trace(go.Bar(name="حصة الإيراد (GMV)", x=seg.index, y=seg["pct_gmv"] * 100,
                         marker_color=C["navy"], text=[fmt_pct(v) for v in seg["pct_gmv"]],
                         textposition="outside"))
    style(fig, 380)
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(ticksuffix="%", range=[0, max(seg["pct_orders"].max(), seg["pct_gmv"].max()) * 100 * 1.18])
    return fig


def fig_categories(items, n=10):
    cat = items.groupby("category").agg(gmv=("item_revenue", "sum"), rev=("review", "mean"))
    cat = cat.sort_values("gmv", ascending=False).head(n).sort_values("gmv")
    fig = go.Figure(go.Bar(
        x=cat["gmv"], y=cat.index, orientation="h",
        marker=dict(color=cat["rev"], colorscale=[[0, C["red"]], [0.5, C["amber"]], [1, C["teal"]]],
                    cmin=float(cat["rev"].min()), cmax=float(cat["rev"].max()), showscale=True,
                    colorbar=dict(title=dict(text="التقييم"), thickness=10, len=0.8)),
        text=[fmt_money(v) for v in cat["gmv"]], textposition="outside", customdata=cat["rev"],
        hovertemplate="%{y}<br>GMV: R$ %{x:,.0f}<br>متوسط التقييم: %{customdata:.2f}<extra></extra>"))
    style(fig, 380)
    fig.update_xaxes(range=[0, cat["gmv"].max() * 1.25], tickformat=",.2s")
    fig.update_yaxes(showgrid=False)
    return fig


def fig_payment(orders):
    p = orders["pay_type"].value_counts()
    fig = go.Figure(go.Pie(
        labels=[PAY_LABELS.get(x, x) for x in p.index], values=p.values, hole=0.62, sort=False,
        marker=dict(colors=[C["navy"], C["blue"], C["amber"], C["teal"], C["slate"]][:len(p)]),
        textinfo="percent", hovertemplate="%{label}: %{value:,.0f} طلب<extra></extra>"))
    style(fig, 380)
    fig.update_layout(legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"))
    return fig


def fig_review_dist(orders):
    s = orders["review"].round().value_counts(normalize=True).reindex([1, 2, 3, 4, 5]).fillna(0)
    fig = go.Figure(go.Bar(
        x=[f"{i} ★" for i in s.index], y=s.values * 100, marker_color=[SCORE_COLORS[i] for i in s.index],
        text=[fmt_pct(v) for v in s.values], textposition="outside",
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    style(fig, 330)
    fig.update_yaxes(ticksuffix="%", range=[0, s.max() * 100 * 1.2])
    return fig


def fig_days_by_score(orders):
    d = orders.dropna(subset=["review", "delivery_days"])
    g = d.groupby(d["review"].round())["delivery_days"].mean().reindex([1, 2, 3, 4, 5])
    fig = go.Figure(go.Bar(
        x=[f"{i} ★" for i in g.index], y=g.values, marker_color=[SCORE_COLORS[i] for i in g.index],
        text=[fmt_num(v) for v in g.values], textposition="outside",
        hovertemplate="%{x}: %{y:.1f} يوم<extra></extra>"))
    style(fig, 330)
    fig.update_yaxes(ticksuffix=" يوم", range=[0, np.nanmax(g.values) * 1.2])
    return fig


def fig_bad_by_bucket(orders):
    g = orders.groupby("delivery_bucket", observed=True).agg(bad=("is_bad", "mean"), n=("order_id", "size"))
    fig = go.Figure(go.Bar(
        x=[str(i) for i in g.index], y=g["bad"] * 100, customdata=g["n"],
        marker=dict(color=g["bad"] * 100, colorscale=[[0, C["teal"]], [0.5, C["amber"]], [1, C["red"]]]),
        text=[fmt_pct(v) for v in g["bad"]], textposition="outside",
        hovertemplate="%{x}<br>تقييمات سلبية: %{y:.1f}%<br>الطلبات: %{customdata:,.0f}<extra></extra>"))
    style(fig, 330)
    fig.update_yaxes(ticksuffix="%", range=[0, g["bad"].max() * 100 * 1.2])
    return fig


def fig_state_days(orders, n=12):
    g = orders.groupby("cust_state").agg(days=("delivery_days", "mean"), n=("order_id", "size"))
    g = g.sort_values("n", ascending=False).head(n).sort_values("days")
    fig = go.Figure(go.Bar(
        x=g["days"], y=g.index, orientation="h", customdata=g["n"],
        marker=dict(color=g["days"], colorscale=[[0, C["teal"]], [0.5, C["amber"]], [1, C["red"]]]),
        text=[fmt_num(v) for v in g["days"]], textposition="outside",
        hovertemplate="%{y}<br>متوسط التسليم: %{x:.1f} يوم<br>الطلبات: %{customdata:,.0f}<extra></extra>"))
    style(fig, 400)
    fig.update_xaxes(range=[0, g["days"].max() * 1.2], ticksuffix=" يوم")
    fig.update_yaxes(showgrid=False)
    return fig


def fig_route_heatmap(orders, min_orders=30):
    ss = orders["seller_state"].value_counts().head(8).index.tolist()
    cs = orders["cust_state"].value_counts().head(10).index.tolist()
    sub = orders[orders["seller_state"].isin(ss) & orders["cust_state"].isin(cs)]
    if sub.empty:
        return None
    piv = sub.pivot_table(index="seller_state", columns="cust_state", values="delivery_days", aggfunc="mean")
    cnt = sub.pivot_table(index="seller_state", columns="cust_state", values="order_id", aggfunc="count")
    piv = piv.where(cnt >= min_orders).reindex(index=ss, columns=cs)
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=cs, y=ss, hoverongaps=False, text=np.round(piv.values, 1), texttemplate="%{text}",
        colorscale=[[0, C["teal"]], [0.5, "#FDE68A"], [1, C["red"]]],
        colorbar=dict(title=dict(text="أيام"), thickness=10),
        hovertemplate="البائع: %{y} ← العميل: %{x}<br>%{z:.1f} يوم<extra></extra>"))
    style(fig, 400)
    fig.update_xaxes(title_text="ولاية العميل", side="bottom")
    fig.update_yaxes(title_text="ولاية البائع", autorange="reversed", showgrid=False)
    return fig


def fig_seg_bar(seg, col, title_fmt, pct=False, suffix=""):
    vals = seg[col] * (100 if pct else 1)
    fig = go.Figure(go.Bar(
        x=seg.index, y=vals, marker_color=[SEG_COLORS[s] for s in seg.index],
        text=[title_fmt(v) for v in seg[col]], textposition="outside",
        hovertemplate="%{x}<br>%{y:.2f}<extra></extra>"))
    style(fig, 300)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(ticksuffix=suffix, range=[0, float(vals.max()) * 1.22])
    return fig


def fig_order_value_hist(orders):
    cap = orders["order_total"].quantile(0.99)
    d = orders.loc[orders["order_total"] <= cap, ["order_total", "segment"]]
    fig = px.histogram(d, x="order_total", color="segment", nbins=70, color_discrete_map=SEG_COLORS,
                       category_orders={"segment": SEG_ORDER},
                       labels={"order_total": "قيمة الطلب (R$)", "segment": "الشريحة", "count": "عدد الطلبات"})
    style(fig, 330)
    fig.update_layout(barmode="stack", bargap=0.02)
    fig.update_yaxes(title_text="عدد الطلبات")
    return fig


def fig_pareto(sel):
    x = np.arange(1, len(sel) + 1) / len(sel) * 100
    n80 = sellers_for_share(sel)
    fig = go.Figure(go.Scatter(
        x=x, y=sel["cum_share"] * 100, mode="lines", fill="tozeroy",
        line=dict(color=C["blue"], width=3), fillcolor="rgba(37,99,235,0.12)",
        hovertemplate="أعلى %{x:.1f}% من البائعين ← %{y:.1f}% من الإيراد<extra></extra>"))
    fig.add_hline(y=80, line_dash="dot", line_color=C["red"], annotation_text="80% من الإيراد",
                  annotation_position="bottom right")
    fig.add_vline(x=n80 / len(sel) * 100, line_dash="dot", line_color=C["red"])
    style(fig, 360)
    fig.update_xaxes(title_text="% من البائعين (مرتبين حسب الإيراد)", ticksuffix="%")
    fig.update_yaxes(title_text="% تراكمي من الإيراد", ticksuffix="%", range=[0, 102])
    return fig


def fig_seller_scatter(sel, avg_review, min_items=30):
    d = sel[sel["items"] >= min_items].dropna(subset=["avg_review", "avg_days"])
    if d.empty:
        return None
    fig = px.scatter(
        d, x="avg_days", y="avg_review", size="gmv", size_max=38, opacity=0.65, hover_name="short_id",
        hover_data={"gmv": ":,.0f", "orders": ":,", "avg_days": ":.1f", "avg_review": ":.2f"},
        labels={"avg_days": "متوسط زمن التسليم (يوم)", "avg_review": "متوسط التقييم", "gmv": "GMV",
                "orders": "الطلبات"}, color_discrete_sequence=[C["blue"]])
    fig.add_hline(y=avg_review, line_dash="dot", line_color=C["slate"],
                  annotation_text="متوسط المنصة", annotation_position="top left")
    style(fig, 360)
    fig.update_xaxes(showgrid=True, gridcolor=C["grid"])
    return fig


def fig_seller_states(items, n=8):
    g = items.groupby("seller_state")["item_revenue"].sum().sort_values(ascending=False).head(n).sort_values()
    fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h", marker_color=C["navy"],
                           text=[fmt_money(v) for v in g.values], textposition="outside",
                           hovertemplate="%{y}: R$ %{x:,.0f}<extra></extra>"))
    style(fig, 360)
    fig.update_xaxes(range=[0, g.max() * 1.25], tickformat=",.2s")
    fig.update_yaxes(showgrid=False)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 6) واجهة المستخدم
# ══════════════════════════════════════════════════════════════════════
def get_data_source():
    path = find_data_file()
    with st.sidebar.expander("📁 مصدر البيانات", expanded=path is None):
        up = st.file_uploader("ارفع ملف CSV (بدون صف عناوين)", type=["csv"])
        if path:
            st.caption(f"الملف الحالي: {Path(path).name}")
    if up is not None:
        return up.getvalue()
    return path


def apply_filters(items, orders):
    sb = st.sidebar
    sb.markdown("### 🎛️ الفلاتر")
    dmin, dmax = orders["purchase_ts"].min().date(), orders["purchase_ts"].max().date()
    dr = sb.date_input("الفترة الزمنية", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    if isinstance(dr, (tuple, list)):
        start, end = (dr[0], dr[1]) if len(dr) == 2 else (dr[0], dr[0]) if len(dr) == 1 else (dmin, dmax)
    else:
        start = end = dr
    state_opts = orders["cust_state"].value_counts().index.tolist()
    states = sb.multiselect("ولاية العميل", state_opts, placeholder="الكل")
    segs = sb.multiselect("شريحة قيمة الطلب", SEG_ORDER, placeholder="الكل")
    cat_opts = items.groupby("category")["item_revenue"].sum().sort_values(ascending=False).index.tolist()
    cats = sb.multiselect("فئة المنتج", cat_opts, placeholder="الكل")

    o = orders[(orders["purchase_ts"] >= pd.Timestamp(start)) &
               (orders["purchase_ts"] < pd.Timestamp(end) + pd.Timedelta(days=1))]
    if states:
        o = o[o["cust_state"].isin(states)]
    if segs:
        o = o[o["segment"].isin(segs)]
    i = items[items["order_id"].isin(o["order_id"])]
    if cats:
        i = i[i["category"].isin(cats)]
        o = o[o["order_id"].isin(i["order_id"])]
    return i, o


def sidebar_notes(dq):
    with st.sidebar.expander("🛠️ ملاحظات هندسة البيانات"):
        st.markdown(
            f"""
- **الصفوف:** {dq['rows']:,} بنداً ← **{dq['orders']:,}** طلباً فريداً · **{dq['sellers']:,}** بائعاً.
- **الإيراد** يُحسب من البنود (`price + freight`)، و**المدفوعات وعدد الطلبات** من الطلبات الفريدة فقط (لا تضخم).
- **الشرائح** أُعيدت تسميتها لأنها مبنية على *قيمة الطلب الواحد* وليست ولاء/LTV.
- **{dq['unreviewed_rows']:,}** صف بلا تقييم حُوّلت إلى `NaN` واستُبعدت من المتوسطات.
- **{dq['dup_rows']:,}** صف مكرر بالكامل أُبقي (غالباً كميات متعددة)، لغياب `order_item_id`.
- **{dq['missing_category']:,}** صف بلا فئة (صُنّفت «غير مصنّف»)، و**{dq['missing_delivery']}** بلا تاريخ تسليم.
- كل الطلبات `delivered`: لا يمكن قياس الإلغاء/الإرجاع.
            """
        )


def hero(o, i):
    period = f"{o['purchase_ts'].min():%Y-%m-%d} إلى {o['purchase_ts'].max():%Y-%m-%d}"
    md(f"""
    <div class="hero">
      <div class="hero-title">📊 لوحة القيادة التنفيذية — Olist E-Commerce</div>
      <p class="hero-sub">المبيعات · رضا العملاء · اللوجستيات · أداء البائعين — من البيانات إلى القرار</p>
      <span class="chip">🗓️ {period}</span>
      <span class="chip">🧾 {len(o):,} طلب</span>
      <span class="chip">📦 {len(i):,} بند</span>
      <span class="chip">🏪 {i['seller_id'].nunique():,} بائع</span>
      <span class="chip">💡 القصة والتوصيات في التبويب الأخير</span>
    </div>""")


def render_kpis(k):
    r1 = st.columns(4)
    r1[0].markdown(kpi_card("💰 إجمالي المبيعات GMV", fmt_money(k["gmv"]),
                            f"السعر + الشحن · {fmt_int(k['n_items'])} بند", C["blue"]), unsafe_allow_html=True)
    r1[1].markdown(kpi_card("🧾 عدد الطلبات الفريدة", fmt_int(k["n_orders"]),
                            f"{k['items_per_order']:.2f} بند لكل طلب", C["navy"]), unsafe_allow_html=True)
    r1[2].markdown(kpi_card("🛒 متوسط قيمة الطلب AOV", fmt_money(k["aov"]),
                            f"إجمالي المدفوعات {fmt_money(k['payments'])}", C["teal"]), unsafe_allow_html=True)
    r1[3].markdown(kpi_card("🚚 حصة الشحن من GMV", fmt_pct(k["freight_share"]),
                            f"متوسط الشحن {fmt_money(k['freight_per_item'])} للبند", C["amber"]),
                   unsafe_allow_html=True)
    r2 = st.columns(4)
    r2[0].markdown(kpi_card("⭐ متوسط التقييم", f"{k['avg_review']:.2f} / 5",
                            "الطلبات المُقيَّمة فقط", C["teal"]), unsafe_allow_html=True)
    r2[1].markdown(kpi_card("😡 نسبة التقييمات السلبية", fmt_pct(k["bad_rate"]),
                            "تقييم 1–2 من المُقيَّمة", C["red"]), unsafe_allow_html=True)
    r2[2].markdown(kpi_card("⏱️ متوسط زمن التسليم", f"{k['avg_days']:.1f} يوم",
                            f"الوسيط {k['med_days']:.1f} · P90 {k['p90_days']:.1f} يوم", C["blue"]),
                   unsafe_allow_html=True)
    r2[3].markdown(kpi_card("⚠️ الطلبات غير المُقيَّمة", fmt_pct(k["unreviewed_rate"], 2),
                            f"{k['unreviewed_n']:,} طلب (استُبعدت من المتوسط)", C["slate"]),
                   unsafe_allow_html=True)


# ─── التبويب 1: نظرة عامة ───────────────────────────────────────────
def tab_overview(items, orders):
    k = compute_kpis(items, orders)
    seg = segment_summary(items, orders)
    section("المؤشرات الرئيسية", "محسوبة على الطلبات الفريدة والبنود بدون تضخم")
    render_kpis(k)

    section("اتجاه المبيعات وتركّز الإيراد")
    c1, c2 = st.columns([3, 2])
    with c1, st.container(border=True):
        chart_header("الاتجاه الشهري: GMV مقابل عدد الطلبات",
                     "استُبعدت الأشهر الجزئية (أقل من 30 طلباً)")
        show(fig_monthly(items, orders), "monthly")
    with c2, st.container(border=True):
        chart_header("حصة الطلبات مقابل حصة الإيراد", "حسب شريحة قيمة الطلب الواحد")
        if len(seg):
            show(fig_segment_share(seg), "segshare")
            if SEG_TOP in seg.index:
                t = seg.loc[SEG_TOP]
                insight(f"الشريحة الكبرى = <b>{fmt_pct(t['pct_orders'])}</b> من الطلبات لكنها "
                        f"<b>{fmt_pct(t['pct_gmv'])}</b> من الإيراد.")

    c3, c4 = st.columns([3, 2])
    with c3, st.container(border=True):
        chart_header("أعلى 10 فئات إيراداً", "اللون = متوسط التقييم (أحمر منخفض ← أخضر مرتفع)")
        show(fig_categories(items), "cats")
    with c4, st.container(border=True):
        chart_header("طرق الدفع", "حصة الطلبات (الطريقة الأولى للطلب)")
        show(fig_payment(orders), "pay")


# ─── التبويب 2: التسليم والرضا ──────────────────────────────────────
def tab_delivery(items, orders):
    section("التسليم هو محرّك الرضا", "العلاقة بين زمن التسليم وتقييم العميل")
    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        chart_header("توزيع درجات التقييم", "حصة الطلبات المُقيَّمة")
        show(fig_review_dist(orders), "revdist")
    with c2, st.container(border=True):
        chart_header("متوسط أيام التسليم لكل درجة تقييم", "كلما تأخر الطلب تراجع التقييم")
        show(fig_days_by_score(orders), "daysscore")
    with c3, st.container(border=True):
        chart_header("نسبة التقييمات السلبية حسب فئة الزمن", "1–2 نجمة من الطلبات المُقيَّمة")
        show(fig_bad_by_bucket(orders), "badbucket")

    section("الجغرافيا واللوجستيات", "أين يتأخر الطلب ولماذا؟")
    c4, c5 = st.columns(2)
    with c4, st.container(border=True):
        chart_header("متوسط التسليم حسب ولاية العميل", "أعلى 12 ولاية من حيث عدد الطلبات")
        show(fig_state_days(orders), "statedays")
    with c5, st.container(border=True):
        chart_header("خريطة حرارية: مسار الشحن (بائع ← عميل)", "متوسط أيام التسليم · مسارات ≥ 30 طلباً")
        hm = fig_route_heatmap(orders)
        if hm is None:
            st.info("لا توجد بيانات كافية للمسارات ضمن الفلاتر الحالية.")
        else:
            show(hm, "heat")

    c6, c7 = st.columns(2)
    with c6:
        chart_header("داخل الولاية مقابل عبر الولايات")
        by = orders.groupby("is_same_state").agg(n=("order_id", "size"), days=("delivery_days", "mean"),
                                                 rev=("review", "mean"), bad=("is_bad", "mean"))
        fr = items.groupby("is_same_state")["freight"].mean()
        rows = []
        for flag, name in [(True, "داخل نفس الولاية"), (False, "عبر الولايات")]:
            if flag in by.index:
                r = by.loc[flag]
                rows.append({"النوع": name, "حصة الطلبات": fmt_pct(r["n"] / by["n"].sum()),
                             "متوسط التسليم (يوم)": fmt_num(r["days"]), "متوسط الشحن للبند": fmt_money(fr.get(flag)),
                             "التقييم": rating_badge(r["rev"]), "سلبي": fmt_pct(r["bad"])})
        if rows:
            md(html_table(pd.DataFrame(rows)))
    with c7:
        chart_header("أداء المناطق البرازيلية (حسب موقع العميل)")
        g = orders.groupby("cust_region").agg(n=("order_id", "size"), days=("delivery_days", "mean"),
                                              rev=("review", "mean"), bad=("is_bad", "mean")).sort_values("n", ascending=False)
        t = pd.DataFrame({"المنطقة": g.index, "الطلبات": [fmt_int(v) for v in g["n"]],
                          "الحصة": [fmt_pct(v / g["n"].sum()) for v in g["n"]],
                          "التسليم (يوم)": [fmt_num(v) for v in g["days"]],
                          "التقييم": [rating_badge(v) for v in g["rev"]],
                          "سلبي": [fmt_pct(v) for v in g["bad"]]})
        md(html_table(t))


# ─── التبويب 3: الشرائح ─────────────────────────────────────────────
def tab_segments(items, orders):
    section("تقسيم الطلبات حسب قيمة الطلب الواحد (Order Value Segmentation)")
    insight("التصنيف الأصلي (VIP / Gold) مبني على <b>قيمة الطلب الواحد</b> وليس على ولاء العميل أو LTV "
            "(لا يوجد customer_id). لذلك أُعيدت تسميته: <b>كبرى ≥ 1,000</b> · <b>مرتفعة 500–1,000</b> · "
            "<b>قياسية &lt; 500</b> (R$).")
    seg = segment_summary(items, orders)
    if seg.empty:
        st.info("لا توجد بيانات ضمن الفلاتر الحالية.")
        return
    t = pd.DataFrame({
        "الشريحة": seg.index, "الطلبات": [fmt_int(v) for v in seg["orders"]],
        "% الطلبات": [fmt_pct(v) for v in seg["pct_orders"]], "GMV": [fmt_money(v) for v in seg["gmv"]],
        "% GMV": [fmt_pct(v) for v in seg["pct_gmv"]], "AOV": [fmt_money(v) for v in seg["aov"]],
        "شحن/طلب": [fmt_money(v) for v in seg["freight_order"]],
        "التسليم (يوم)": [fmt_num(v) for v in seg["days"]],
        "التقييم": [rating_badge(v) for v in seg["avg_review"]],
        "% سلبي": [fmt_pct(v) for v in seg["bad"]],
    })
    md(html_table(t))
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        chart_header("نسبة التقييمات السلبية", "الأعلى = الأسوأ")
        show(fig_seg_bar(seg, "bad", fmt_pct, pct=True, suffix="%"), "seg_bad")
    with c2, st.container(border=True):
        chart_header("متوسط زمن التسليم (يوم)")
        show(fig_seg_bar(seg, "days", fmt_num), "seg_days")
    with c3, st.container(border=True):
        chart_header("الشحن كنسبة من سعر المنتج")
        show(fig_seg_bar(seg, "freight_pct_price", fmt_pct, pct=True, suffix="%"), "seg_frt")
    with st.container(border=True):
        chart_header("توزيع قيم الطلبات", "مقصوص عند المئين 99 لوضوح الرسم")
        show(fig_order_value_hist(orders), "hist")


# ─── التبويب 4: البائعون ────────────────────────────────────────────
def tab_sellers(items, orders):
    sel = seller_table(items)
    if sel.empty:
        st.info("لا توجد بيانات ضمن الفلاتر الحالية.")
        return
    n80 = sellers_for_share(sel)
    sp_share = (items["seller_state"] == "SP").mean()
    section("تركّز الإيراد وجودة البائعين")
    c = st.columns(4)
    c[0].markdown(kpi_card("🏪 البائعون النشطون", fmt_int(len(sel)), "بائع لديه مبيعات", C["navy"]), unsafe_allow_html=True)
    c[1].markdown(kpi_card("📌 بائعو 80% من الإيراد", fmt_int(n80), f"{fmt_pct(n80 / len(sel))} من البائعين", C["blue"]), unsafe_allow_html=True)
    c[2].markdown(kpi_card("🔟 حصة أكبر 10 بائعين", fmt_pct(sel.head(10)["share"].sum()), "من GMV", C["teal"]), unsafe_allow_html=True)
    c[3].markdown(kpi_card("📍 بنود من بائعي ساو باولو", fmt_pct(sp_share), "SP — تركّز جغرافي", C["amber"]), unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1, st.container(border=True):
        chart_header("منحنى باريتو لتركّز الإيراد", "كم بائعاً يصنع معظم الإيراد؟")
        show(fig_pareto(sel), "pareto")
    with c2, st.container(border=True):
        chart_header("GMV حسب ولاية البائع", "أعلى 8 ولايات")
        show(fig_seller_states(items), "sellerstates")

    with st.container(border=True):
        chart_header("خريطة الجودة: زمن التسليم مقابل التقييم", "حجم الفقاعة = GMV · بائعون لديهم 30 بنداً فأكثر")
        sc = fig_seller_scatter(sel, orders["review"].mean())
        if sc is None:
            st.info("لا يوجد بائعون بحجم كافٍ ضمن الفلاتر الحالية.")
        else:
            show(sc, "sellerscatter")

    section("أكبر 10 بائعين", "الشارة الحمراء = تقييم أقل من 3.8")
    top = sel.head(10)
    t = pd.DataFrame({
        "#": range(1, len(top) + 1), "البائع": top["short_id"], "المدينة": top["city"], "الولاية": top["state"],
        "GMV": [fmt_money(v) for v in top["gmv"]], "الحصة": [fmt_pct(v) for v in top["share"]],
        "البنود": [fmt_int(v) for v in top["items"]], "التقييم": [rating_badge(v) for v in top["avg_review"]],
        "التسليم (يوم)": [fmt_num(v) for v in top["avg_days"]], "% سلبي": [fmt_pct(v) for v in top["bad"]],
    })
    md(html_table(t))


# ─── التبويب 5: القصة والتوصيات ──────────────────────────────────────
def tab_story(items_all, orders_all):
    S = compute_story(items_all, orders_all)
    k, ws = S["k"], S["worst_top"]
    insight("أرقام هذا التبويب محسوبة على <b>كامل البيانات</b> ولا تتأثر بالفلاتر الجانبية.")

    md(f"""
    <div class="story-hero">
      <div class="h">القصة في جملة واحدة: أغلى عملائنا هم الأقل رضاً، والتأخير الجغرافي هو العدو الأول للرضا.</div>
      <div class="p">حققت المنصة <b>{fmt_money(k['gmv'])}</b> من <b>{fmt_int(k['n_orders'])}</b> طلباً بمتوسط تقييم
      <b>{k['avg_review']:.2f}/5</b> — رقم يبدو صحياً. لكن خلفه ثلاث حقائق: الطلبات الأكبر قيمة
      (<b>{fmt_pct(S['prem_orders'])}</b> من الطلبات، <b>{fmt_pct(S['prem_gmv'])}</b> من الإيراد) هي الأكثر غضباً،
      والطلب الذي يعبر ولايات يتأخر ضعف الطلب المحلي تقريباً، والنمو استقر بعد ذروة {S['peak_label']}.</div>
    </div>""")

    section("أين تكمن المشكلة؟", "ثلاثة اكتشافات مبنية على الأرقام")
    f1, f2, f3 = st.columns(3)
    f1.markdown(_compact(f"""
    <div class="finding"><span class="n">1</span>
      <div class="t">المفارقة: الأكثر دفعاً هم الأقل رضاً</div>
      <div class="b"><span class="big red">{fmt_pct(S['bad_top'])}</span> تقييمات سلبية في الشريحة الكبرى
      و<b>{fmt_pct(S['bad_high'])}</b> في المرتفعة، مقابل <b>{fmt_pct(S['bad_std'])}</b> فقط في القياسية.
      متوسط تقييم الكبرى <b>{S['rev_top']:.2f}</b> مقابل <b>{S['rev_std']:.2f}</b>.
      وارتبط <b>{fmt_money(S['risk_gmv'])}</b> من إيراد الشريحتين بتقييمات 1–2.</div></div>"""), unsafe_allow_html=True)
    f2.markdown(_compact(f"""
    <div class="finding"><span class="n">2</span>
      <div class="t">التأخير يقتل الرضا</div>
      <div class="b">طلبات 5 نجوم تصل في <b>{S['days_5']:.1f}</b> يوماً، وطلبات النجمة الواحدة في
      <b class="red">{S['days_1']:.1f}</b> يوماً. التقييمات السلبية تقفز من <b>{fmt_pct(S['bad_fast'])}</b> (أقل من 7 أيام)
      إلى <b class="red">{fmt_pct(S['bad_slow'])}</b> (أكثر من 21 يوماً) — و{fmt_pct(S['slow_share'])} من الطلبات في الفئة الأبطأ.</div></div>"""), unsafe_allow_html=True)
    f3.markdown(_compact(f"""
    <div class="finding"><span class="n">3</span>
      <div class="t">الجغرافيا تصنع التأخير</div>
      <div class="b">{fmt_pct(S['cross_share'],0)} من الطلبات تعبر الولايات وتصل في <b class="red">{S['days_cross']:.1f}</b> يوماً
      مقابل <b class="teal">{S['days_same']:.1f}</b> يوماً داخل الولاية، بشحن {fmt_money(S['frt_cross'])} مقابل {fmt_money(S['frt_same'])} للبند.
      متوسط ساو باولو {S['sp_days']:.1f} يوماً فقط.</div></div>"""), unsafe_allow_html=True)

    st.write("")
    weak = "أقوى بكثير" if abs(S["corr_days"]) > abs(S["corr_freight"]) else "قريب"
    md(f"""
    <div class="finding"><div class="t">📎 أدلة داعمة</div><div class="b">
    • <b>السرعة لا السعر:</b> ارتباط زمن التسليم بالتقييم <b>{S['corr_days']:.2f}</b>، أما نسبة الشحن للسعر فارتباطها
    <b>{S['corr_freight']:.2f}</b> — الأثر {weak} لصالح السرعة.<br>
    • <b>النمو مستقر:</b> بعد ذروة {S['peak_label']} ({fmt_int(S['peak_orders'])} طلب) استقر الحجم عند ≈{fmt_int(S['last6_avg'])}
    طلب شهرياً (≈{fmt_pct(S['last6_ratio'],0)} من الذروة).<br>
    • <b>تركّز البائعين:</b> {fmt_int(S['n80'])} بائعاً ({fmt_pct(S['n80_pct'])}) يصنعون 80% من الإيراد، وأكبر 10 بائعين = {fmt_pct(S['top10_share'])} من GMV.<br>
    • <b>ضعف جودة بين الكبار:</b> البائع <b>{ws['short_id']}</b> (رقم {S['worst_top_rank']} في الإيراد، حصة {fmt_pct(ws['share'])})
    تقييمه <b class="red">{ws['avg_review']:.2f}</b> مقابل {k['avg_review']:.2f} متوسط المنصة.</div></div>""")

    section("التوصيات التنفيذية", "مرتبة حسب الأولوية ومربوطة بأرقام فعلية")
    slow = S["slow_states"]
    slow_txt = "، ".join(f"{s} ({r['days']:.1f} يوم)" for s, r in slow.iterrows()) or "—"
    cw, cb = S["cat_worst_row"], S["cat_best_row"]
    recs = [
        ("hi", "عاجل", "مسار خدمة مميز للطلبات الكبرى (+500)",
         f"الشريحتان تمثلان {fmt_pct(S['prem_orders'])} من الطلبات و{fmt_pct(S['prem_gmv'])} من GMV، لكن السلبي فيهما "
         f"{fmt_pct(S['bad_high'])}–{fmt_pct(S['bad_top'])} مقابل {fmt_pct(S['bad_std'])}. طبّق أولوية تجهيز وشحن وتتبعاً استباقياً "
         f"وتواصلاً فورياً عند أي تأخير.",
         f"الهدف: خفض السلبي في الشريحة الكبرى إلى {fmt_pct(S['bad_std'])} وحماية ≈{fmt_money(S['risk_gmv'])} من الإيراد."),
        ("hi", "عاجل", "تقليص الشحن عبر الولايات بمخزون/مراكز إقليمية",
         f"{fmt_pct(S['cross_share'],0)} من الطلبات تعبر الولايات وتتأخر ({S['days_cross']:.1f} مقابل {S['days_same']:.1f} يوماً). "
         f"الولايات الأعلى كلفة زمنية (حجم × تأخير): {slow_txt}. حفّز البائعين على التخزين الإقليمي أو أنشئ مراكز تجميع قربها.",
         f"الهدف: خفض حصة الطلبات فوق 21 يوماً من {fmt_pct(S['slow_share'])} إلى النصف."),
        ("hi", "عاجل", "برنامج جودة للبائعين الكبار",
         f"البائع {ws['short_id']} ضمن أكبر 10 بائعين (حصة {fmt_pct(ws['share'])}) وتقييمه {ws['avg_review']:.2f} فقط. "
         f"ضع خطة تحسين 90 يوماً بمؤشرات (تقييم، زمن تسليم) وشروط إيقاف. وراقب مخاطر التركّز: "
         f"{fmt_int(S['n80'])} بائعاً يصنعون 80% من الإيراد.",
         "الهدف: لا بائع ضمن أكبر 20 بتقييم أقل من متوسط المنصة بأكثر من 0.3."),
        ("md", "متوسط", "إنذار تأخير وتعويض تلقائي",
         f"الطلبات فوق 21 يوماً تسجل {fmt_pct(S['bad_slow'])} سلبياً مقابل {fmt_pct(S['bad_fast'])} لمن وصلهم الطلب خلال أقل من 7 أيام. "
         f"أطلق تنبيهاً عند اليوم 14 يتبعه اعتذار/قسيمة تلقائية، واعتمد P90 (حالياً {k['p90_days']:.1f} يوماً) مؤشراً أسبوعياً.",
         "الهدف: خفض P90 وتحويل جزء من التقييمات السلبية المحتملة إلى محايدة."),
        ("md", "متوسط", "السرعة قبل دعم الشحن",
         f"ارتباط التقييم بزمن التسليم ({S['corr_days']:.2f}) يفوق ارتباطه بنسبة الشحن ({S['corr_freight']:.2f}). "
         f"استثمر أولاً في السرعة والوعد الواقعي للتسليم؛ خفض رسوم الشحن وحده لن يحسّن الرضا بشكل ملموس.",
         "الهدف: ربط ميزانية الشحن بمؤشر الزمن لا بالسعر."),
        ("md", "متوسط", "إعادة تحريك النمو عبر الفئات الأعلى رضاً",
         f"استقر الحجم عند ≈{fmt_int(S['last6_avg'])} طلب شهرياً. ركّز الاستحواذ على فئات مثل «{S['cat_best']}» "
         f"(تقييم {cb['rev']:.2f}، حصة {fmt_pct(cb['share'])})، وراجع موردي/وصف «{S['cat_worst']}» "
         f"(تقييم {cw['rev']:.2f}، حصة {fmt_pct(cw['share'])}).",
         "الهدف: رفع الطلبات الشهرية فوق ذروة " + S["peak_label"] + "."),
        ("md", "متوسط", "تحسين جاهزية البيانات",
         "أضف customer_unique_id (لـ LTV/RFM حقيقي)، وتاريخ التسليم المقدَّر (لقياس التأخير عن الوعد)، وحالة الطلب الكاملة "
         "(الإلغاء/الإرجاع)، و order_item_id و product_id، وأبعاد المنتج ووزنه لتفسير الشحن، وشجّع التقييم "
         f"(غير المُقيَّم حالياً {fmt_pct(k['unreviewed_rate'],2)}).",
         "الهدف: الانتقال من تحليل الطلب الواحد إلى تحليل العميل."),
    ]
    for i, (lvl, tag, title, body, target) in enumerate(recs, 1):
        md(f'<div class="rec {lvl}"><div class="t"><span class="pill {lvl}">{tag}</span>{i}. {title}</div>'
           f'<div class="b">{body}</div><div class="k">🎯 {target}</div></div>')

    section("حدود التحليل")
    md("""<div class="finding"><div class="b">
    • كل الطلبات بحالة <b>delivered</b> ← لا يمكن قياس الإلغاء والإرجاع (انحياز للنجاح).<br>
    • لا يوجد <b>customer_id</b> ← الشرائح تعبّر عن قيمة الطلب لا عن قيمة العميل مدى الحياة.<br>
    • لا يوجد تاريخ تسليم مقدَّر ← التأخير يُقاس بالأيام المطلقة لا بالفارق عن الوعد.<br>
    • العلاقات المعروضة <b>ارتباطات</b> وليست إثباتاً سببياً؛ يُنصح باختبار التوصيات تجريبياً (A/B أو Pilot إقليمي).</div></div>""")


# ══════════════════════════════════════════════════════════════════════
# 7) نقطة الدخول
# ══════════════════════════════════════════════════════════════════════
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown("## 📊 Olist Analytics")
    source = get_data_source()
    if source is None:
        st.warning(f"لم يُعثر على الملف **{DATA_FILENAME}**. ضعه بجانب app.py أو ارفعه من الشريط الجانبي.")
        st.stop()

    items_all, orders_all, dq = load_and_prepare(source)
    items, orders = apply_filters(items_all, orders_all)
    sidebar_notes(dq)
    if orders.empty:
        st.warning("لا توجد بيانات ضمن الفلاتر المختارة. جرّب توسيع الفلاتر.")
        st.stop()

    hero(orders, items)
    t1, t2, t3, t4, t5 = st.tabs(["📈 نظرة عامة", "🚚 التسليم والرضا", "🧩 شرائح قيمة الطلب",
                                  "🏪 البائعون", "🧠 القصة والتوصيات"])
    with t1:
        tab_overview(items, orders)
    with t2:
        tab_delivery(items, orders)
    with t3:
        tab_segments(items, orders)
    with t4:
        tab_sellers(items, orders)
    with t5:
        tab_story(items_all, orders_all)
    md('<div class="foot">Olist Executive Dashboard · Streamlit + Plotly</div>')


if __name__ == "__main__":
    main()
