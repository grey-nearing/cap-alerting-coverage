"""Step 01 (Tier 2): Build GADM v4.1 Admin-1 layer, gazetteer, Catalog A, and Catalog B."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely import make_valid
from shapely.errors import GEOSException
from shapely.geometry import box, shape
from shapely.strtree import STRtree

from cap_alerting_coverage.clustering import (
    PopRule,
    cluster_gid1_episodes,
    merge_strictly_nested_catalog_b,
)
from cap_alerting_coverage.config import ISO_UNIFY, LEGACY_EXCLUDED_19, ProjectPaths

ADMIN_STOPWORDS: frozenset[str] = frozenset(
    {
        "province",
        "provinces",
        "state",
        "states",
        "region",
        "regions",
        "department",
        "departments",
        "departement",
        "departamento",
        "county",
        "counties",
        "district",
        "districts",
        "municipality",
        "municipalities",
        "municipio",
        "city",
        "cities",
        "town",
        "metropolitan",
        "autonomous",
        "special",
        "administrative",
        "territory",
        "territories",
        "prefecture",
        "oblast",
        "krai",
        "kray",
        "republic",
        "okrug",
        "kabupaten",
        "kota",
        "regency",
        "regencies",
        "commune",
        "canton",
        "parish",
        "voivodeship",
        "wojewodztwo",
        "judet",
        "governorate",
        "wilaya",
        "wilayah",
        "raion",
        "rayon",
        "shire",
        "borough",
        "division",
        "zone",
        "sector",
        "area",
        "flood",
        "floods",
        "flooding",
        "flash",
        "rain",
        "heavy",
        "monsoon",
        "river",
        "landslide",
        "landslides",
        "displacement",
        "people",
        "affected",
        "rescued",
        "fatalities",
        "deaths",
        "evacuated",
        "injured",
        "missing",
        "and",
        "the",
        "of",
        "in",
        "del",
        "de",
        "la",
        "el",
        "los",
        "las",
        "da",
        "do",
        "dos",
        "das",
        "north",
        "south",
        "east",
        "west",
        "central",
        "northern",
        "southern",
        "eastern",
        "western",
        "greater",
        "new",
        "upper",
        "lower",
        "mount",
        "saint",
    }
)

CJK_STOPWORDS: frozenset[str] = frozenset(
    {
        "气象",
        "预警",
        "暴雨",
        "洪水",
        "内涝",
        "大风",
        "雷电",
        "冰雹",
        "高温",
        "干旱",
        "台风",
        "中国",
        "人民",
        "政府",
        "发布",
        "中心",
        "预计",
        "今天",
        "明天",
        "地区",
        "部分",
        "出现",
        "可能",
        "影响",
        "天气",
        "降水",
        "降雨",
        "山洪",
        "灾害",
        "蓝色",
        "黄色",
        "橙色",
        "红色",
        "信号",
        "注意",
        "防范",
    }
)

VALID_HUMAN_UNITS: frozenset[str] = frozenset(
    {
        "people",
        "person",
        "persons",
        "individuals",
        "households",
        "household",
        "families",
        "family",
    }
)


def norm_ascii(s: str | None) -> str:
    """Fold diacritics and punctuation to lowercase space-normalized ASCII."""
    if not s or not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[\'’`\-_/.,;:()\[\]{}]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def strip_admin_words(norm_s: str) -> str:
    """Remove generic administrative/hazard stopwords from a normalized string."""
    return " ".join(t for t in norm_s.split() if t not in ADMIN_STOPWORDS).strip()


def split_variants(val: str | None) -> list[str]:
    """Split pipe/semicolon/slash-delimited GADM variant name fields."""
    if not val or not isinstance(val, str) or val in ("NA", "None", "nan"):
        return []
    return [p.strip() for p in re.split(r"[|;/]", val) if p.strip() and p.strip() not in ("NA", "None", "nan")]


def compile_country_matchers(gaz: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Precompile boundary-aware regex and n-gram gazetteer matchers per ISO-3 country."""
    matchers: dict[str, dict[str, Any]] = {}
    adm3_alias = gaz.get("adm3_alias", {})
    adm3_native = gaz.get("adm3_native", {})
    all_isos = set(gaz["adm1_alias"]) | set(gaz["adm2_alias"]) | set(adm3_alias) | set(gaz["single_gid1"])
    for iso in all_isos:
        a1 = gaz["adm1_alias"].get(iso, {})
        a2 = gaz["adm2_alias"].get(iso, {})
        a3 = adm3_alias.get(iso, {})
        nat1 = gaz["adm1_native"].get(iso, {})
        nat2 = gaz["adm2_native"].get(iso, {})
        nat3 = adm3_native.get(iso, {})
        phrases_1 = sorted([k for k in a1 if len(k) >= 4], key=len, reverse=True)
        short_1 = [k for k in a1 if len(k) == 3 and k not in ADMIN_STOPWORDS]
        phrases_2 = sorted([k for k in a2 if len(k) >= 4], key=len, reverse=True)
        re1 = (
            re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases_1 + short_1) + r")\b")
            if (phrases_1 or short_1)
            else None
        )
        re2 = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases_2) + r")\b") if phrases_2 else None
        nat_all = sorted(list(nat1.keys()) + list(nat2.keys()), key=len, reverse=True)
        re_nat = re.compile("(?:" + "|".join(re.escape(p) for p in nat_all) + ")") if nat_all else None
        nat3_lens = sorted({len(k) for k in nat3 if 2 <= len(k) <= 8})
        matchers[iso] = {
            "a1": a1,
            "a2": a2,
            "a3": a3,
            "nat1": nat1,
            "nat2": nat2,
            "nat3": nat3,
            "nat3_lens": nat3_lens,
            "re1": re1,
            "re2": re2,
            "re_nat": re_nat,
            "single": gaz["single_gid1"].get(iso),
            "abbr": gaz["abbrev_map"].get(iso, {}),
        }
    return matchers


def match_text_to_gid1(
    iso: str, raw_text: str | None, matchers: dict[str, dict[str, Any]], allow_single: bool = True
) -> list[str]:
    """Resolve free text to GADM Admin-1 ``gid_1`` identifiers for ``iso`` (zero Admin-0 fallback)."""
    m = matchers.get(iso)
    if not m:
        return []
    if allow_single and m["single"]:
        return [m["single"]]
    if not raw_text or not isinstance(raw_text, str):
        return []
    found: set[str] = set()
    nt = norm_ascii(raw_text)
    if nt and m["re1"]:
        for hit in m["re1"].findall(nt):
            found.update(m["a1"].get(hit, ()))
    if nt and m["re2"]:
        for hit in m["re2"].findall(nt):
            g = m["a2"].get(hit)
            if g:
                found.add(g)
    if m["re_nat"]:
        low = raw_text.lower()
        for hit in m["re_nat"].findall(low):
            found.update(m["nat1"].get(hit, ()))
            g2 = m["nat2"].get(hit)
            if g2:
                found.add(g2)
    if nt and m["a3"]:
        toks = nt.split()
        nt_len = len(toks)
        for i in range(nt_len):
            t1 = toks[i]
            if len(t1) >= 4 and t1 in m["a3"]:
                found.add(m["a3"][t1])
            if i + 1 < nt_len:
                t2 = f"{t1} {toks[i + 1]}"
                if t2 in m["a3"]:
                    found.add(m["a3"][t2])
            if i + 2 < nt_len:
                t3 = f"{t1} {toks[i + 1]} {toks[i + 2]}"
                if t3 in m["a3"]:
                    found.add(m["a3"][t3])
    if m["nat3"]:
        low = raw_text.lower()
        n_toks = re.findall(r"[^\s,.;:()/|]+", low)
        for i, t1 in enumerate(n_toks):
            if t1 in m["nat3"]:
                found.add(m["nat3"][t1])
            if i + 1 < len(n_toks):
                t2 = f"{t1} {n_toks[i + 1]}"
                if t2 in m["nat3"]:
                    found.add(m["nat3"][t2])
        if m["nat3_lens"]:
            for run in re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]{2,}", low):
                rlen = len(run)
                for length in m["nat3_lens"]:
                    if length > rlen:
                        break
                    for start in range(rlen - length + 1):
                        sub = run[start : start + length]
                        if sub in m["nat3"]:
                            found.add(m["nat3"][sub])
    if not found and m["abbr"]:
        for tok in re.findall(r"(?:^|[,;\s(])([A-Z]{2,3})(?=$|[,;\s):])", raw_text):
            if tok in m["abbr"]:
                found.add(m["abbr"][tok])
    return sorted(found)


def is_valid_human_montandon_impact(category: str, impact_type: str, bracket_unit: str) -> bool:
    """Validate that a Montandon impact record counts people/households rather than money or assets."""
    icat = (category or "").strip().lower()
    it = (impact_type or "").strip().lower()
    bu = (bracket_unit or "").strip().lower()
    is_bad_type = any(tok in it for tok in ("cost", "damage", "damaged", "destroyed"))
    return (icat == "people") and (not is_bad_type) and (not bu or bu in VALID_HUMAN_UNITS)


def build_gadm_and_gazetteer(gpkg_path: Path, out_dir: Path) -> tuple[gpd.GeoDataFrame, dict[str, Any], dict[str, Any]]:
    """Build GADM v4.1 Admin-1 parquet and multilingual ADM_1/2/3 -> GID_1 gazetteer JSON."""
    import pyogrio

    out_dir.mkdir(parents=True, exist_ok=True)
    adm1_parquet = out_dir / "gadm41_adm1.parquet"
    gaz_path = out_dir / "gadm41_gazetteer.json"
    if adm1_parquet.exists() and gaz_path.exists():
        adm1_out = gpd.read_parquet(adm1_parquet)
        adm1_out["geometry"] = [make_valid(g) for g in adm1_out.geometry]
        gaz = json.loads(gaz_path.read_text(encoding="utf-8"))
        return adm1_out, gaz, gaz["c0_cents"]

    gdf1 = pyogrio.read_dataframe(gpkg_path, layer="ADM_1")
    gdf0 = pyogrio.read_dataframe(gpkg_path, layer="ADM_0")
    gdf1["country_iso3"] = gdf1["GID_0"].replace(ISO_UNIFY)
    gdf1.loc[gdf1["GID_1"] == "CHN.HKG", "country_iso3"] = "HKG"
    gdf1.loc[gdf1["GID_1"] == "CHN.MAC", "country_iso3"] = "MAC"

    mdv0 = gdf0[gdf0["GID_0"] == "MDV"].copy()
    if not mdv0.empty:
        mdv_row = gpd.GeoDataFrame(
            [
                {
                    "GID_0": "MDV",
                    "COUNTRY": "Maldives",
                    "GID_1": "MDV.0_1",
                    "NAME_1": "Maldives",
                    "VARNAME_1": "",
                    "NL_NAME_1": "",
                    "TYPE_1": "Country",
                    "ENGTYPE_1": "Country",
                    "CC_1": "",
                    "HASC_1": "",
                    "ISO_1": "",
                    "country_iso3": "MDV",
                    "geometry": mdv0.iloc[0]["geometry"],
                }
            ],
            crs=gdf1.crs,
        )
        gdf1 = pd.concat([gdf1, mdv_row], ignore_index=True)

    ab9_row = gpd.GeoDataFrame(
        [
            {
                "GID_0": "AB9",
                "COUNTRY": "Abyei Area",
                "GID_1": "AB9.0_1",
                "NAME_1": "Abyei",
                "VARNAME_1": "Abyei Area",
                "NL_NAME_1": "",
                "TYPE_1": "Special Area",
                "ENGTYPE_1": "Special Area",
                "CC_1": "",
                "HASC_1": "",
                "ISO_1": "",
                "country_iso3": "AB9",
                "geometry": box(27.9, 9.3, 29.0, 10.2),
            }
        ],
        crs=gdf1.crs,
    )
    gdf1 = gpd.GeoDataFrame(pd.concat([gdf1, ab9_row], ignore_index=True), geometry="geometry", crs="EPSG:4326")

    gdf1["area_deg2"] = gdf1.geometry.area
    gdf1["country_area_deg2"] = gdf1["country_iso3"].map(gdf1.groupby("country_iso3")["area_deg2"].sum().to_dict())
    gdf1["n_units_in_country"] = gdf1["country_iso3"].map(gdf1.groupby("country_iso3")["GID_1"].count().to_dict())

    simp_geoms = []
    for geom in gdf1.geometry:
        g_s = geom.simplify(0.002, preserve_topology=True)
        if g_s.is_empty or not g_s.is_valid:
            g_s = geom.buffer(0)
        simp_geoms.append(g_s)
    gdf1["geometry"] = simp_geoms
    rep_pts = gdf1.geometry.representative_point()
    gdf1["rep_lon"] = rep_pts.x
    gdf1["rep_lat"] = rep_pts.y

    gdf0["country_iso3"] = gdf0["GID_0"].replace(ISO_UNIFY)
    gdf0_diss = gdf0.dissolve(by="country_iso3").reset_index()
    c0_cents = {}
    for _, r in gdf0_diss.iterrows():
        c = r["geometry"].centroid
        rp = r["geometry"].representative_point()
        c0_cents[r["country_iso3"]] = [[float(c.x), float(c.y)], [float(rp.x), float(rp.y)]]

    keep_cols = [
        "GID_1",
        "GID_0",
        "country_iso3",
        "COUNTRY",
        "NAME_1",
        "VARNAME_1",
        "NL_NAME_1",
        "TYPE_1",
        "ENGTYPE_1",
        "HASC_1",
        "ISO_1",
        "n_units_in_country",
        "area_deg2",
        "country_area_deg2",
        "rep_lon",
        "rep_lat",
        "geometry",
    ]
    adm1_out = gdf1[keep_cols].rename(columns={"GID_1": "gid_1", "NAME_1": "name_1"}).copy()
    adm1_out.to_parquet(adm1_parquet, index=False)

    # Also emit lightweight geometry-free index for Tier-1 public archives
    adm1_out[["gid_1", "country_iso3"]].to_parquet(out_dir / "gadm41_adm1_index.parquet", index=False)

    df2 = pyogrio.read_dataframe(gpkg_path, layer="ADM_2", read_geometry=False)
    gid1_to_iso = dict(zip(adm1_out["gid_1"], adm1_out["country_iso3"]))
    df2["country_iso3"] = df2["GID_1"].map(gid1_to_iso)
    df2 = df2[df2["country_iso3"].notna()].copy()

    country_adm1_alias: dict[str, dict[str, set[str]]] = {}
    country_adm1_native: dict[str, dict[str, set[str]]] = {}
    country_single_gid1: dict[str, str] = {}
    for _, r in adm1_out.iterrows():
        iso, g1 = r["country_iso3"], r["gid_1"]
        if r["n_units_in_country"] == 1:
            country_single_gid1[iso] = g1
        d = country_adm1_alias.setdefault(iso, {})
        dn = country_adm1_native.setdefault(iso, {})
        for c_raw in [r["name_1"]] + split_variants(r["VARNAME_1"]):
            na = norm_ascii(c_raw)
            if len(na) >= 3:
                d.setdefault(na, set()).add(g1)
            sa = strip_admin_words(na)
            if len(sa) >= 3:
                d.setdefault(sa, set()).add(g1)
        nat_cands = split_variants(r["NL_NAME_1"]) + (
            [r["name_1"]] if any(ord(ch) > 127 for ch in str(r["name_1"])) else []
        )
        for nl_raw in nat_cands:
            nl_s = re.sub(r"\s+", " ", str(nl_raw)).strip()
            if len(nl_s) >= 2:
                dn.setdefault(nl_s.lower(), set()).add(g1)
                nl_short = re.sub(r"(特别行政区|壮族自治区|回族自治区|维吾尔自治区|自治区|省|市)$", "", nl_s)
                if len(nl_short) >= 2 and nl_short != nl_s:
                    dn.setdefault(nl_short.lower(), set()).add(g1)

    adm2_tmp: dict[tuple[str, str], set[str]] = {}
    adm2_nat_tmp: dict[tuple[str, str], set[str]] = {}
    for _, r in df2.iterrows():
        iso, g1 = r["country_iso3"], r["GID_1"]
        for c_raw in [r["NAME_2"]] + split_variants(r["VARNAME_2"]):
            na = norm_ascii(c_raw)
            if len(na) >= 4:
                adm2_tmp.setdefault((iso, na), set()).add(g1)
            sa = strip_admin_words(na)
            if len(sa) >= 4:
                adm2_tmp.setdefault((iso, sa), set()).add(g1)
        for nl_raw in split_variants(r["NL_NAME_2"]):
            if any(ord(ch) > 127 for ch in nl_raw):
                nl_short = re.sub(r"(市|区|县|州|盟)$", "", nl_raw.strip())
                for tok in {nl_raw.strip().lower(), nl_short.lower()}:
                    if len(tok) >= 2:
                        adm2_nat_tmp.setdefault((iso, tok), set()).add(g1)

    country_adm2_alias: dict[str, dict[str, str]] = {}
    for (iso, al), gset in adm2_tmp.items():
        if len(gset) == 1 and al not in country_adm1_alias.get(iso, {}) and al not in ADMIN_STOPWORDS:
            country_adm2_alias.setdefault(iso, {})[al] = next(iter(gset))

    country_adm2_native: dict[str, dict[str, str]] = {}
    for (iso, al), gset in adm2_nat_tmp.items():
        if len(gset) == 1 and al not in country_adm1_native.get(iso, {}):
            country_adm2_native.setdefault(iso, {})[al] = next(iter(gset))

    iso_abbrev_map: dict[str, dict[str, str]] = {}
    for _, r in adm1_out.iterrows():
        iso = r["country_iso3"]
        if iso not in ("USA", "CAN", "AUS", "BRA"):
            continue
        code = str(r["ISO_1"] or r["HASC_1"] or "").strip()
        abbr = code.split("-")[-1].upper() if "-" in code else (code.split(".")[-1].upper() if "." in code else "")
        if len(abbr) == 2 or (iso == "AUS" and abbr in ("NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT")):
            iso_abbrev_map.setdefault(iso, {})[abbr] = r["gid_1"]

    df3 = pyogrio.read_dataframe(gpkg_path, layer="ADM_3", read_geometry=False)
    df3["country_iso3"] = df3["GID_1"].map(gid1_to_iso)
    df3 = df3[df3["country_iso3"].notna()].copy()
    adm3_tmp: dict[tuple[str, str], set[str]] = {}
    adm3_nat_tmp: dict[tuple[str, str], set[str]] = {}
    for _, r in df3.iterrows():
        iso, g1 = r["country_iso3"], r["GID_1"]
        for c_raw in [r["NAME_3"]] + split_variants(r["VARNAME_3"]):
            na = norm_ascii(c_raw)
            if len(na) >= 4:
                adm3_tmp.setdefault((iso, na), set()).add(g1)
            sa = strip_admin_words(na)
            if len(sa) >= 4:
                adm3_tmp.setdefault((iso, sa), set()).add(g1)
        nat_cands = split_variants(r["NL_NAME_3"])
        if r["NAME_3"] and any(ord(ch) > 127 for ch in str(r["NAME_3"])):
            nat_cands.append(str(r["NAME_3"]))
        for nl_raw in nat_cands:
            if any(ord(ch) > 127 for ch in nl_raw):
                s_raw = re.sub(r"\s+", " ", nl_raw.strip())
                s_short = re.sub(r"(自治县|自治旗|特别行政区|街道办事处|地区|市|区|县|镇|乡|旗)$", "", s_raw)
                for tok in {s_raw.lower(), s_short.lower()}:
                    if len(tok) >= 2 and tok not in CJK_STOPWORDS:
                        adm3_nat_tmp.setdefault((iso, tok), set()).add(g1)

    country_adm3_alias: dict[str, dict[str, str]] = {}
    for (iso, al), gset in adm3_tmp.items():
        if (
            len(gset) == 1
            and al not in country_adm1_alias.get(iso, {})
            and al not in country_adm2_alias.get(iso, {})
            and al not in ADMIN_STOPWORDS
        ):
            country_adm3_alias.setdefault(iso, {})[al] = next(iter(gset))

    country_adm3_native: dict[str, dict[str, str]] = {}
    for (iso, al), gset in adm3_nat_tmp.items():
        if (
            len(gset) == 1
            and al not in country_adm1_native.get(iso, {})
            and al not in country_adm2_native.get(iso, {})
            and al not in CJK_STOPWORDS
        ):
            country_adm3_native.setdefault(iso, {})[al] = next(iter(gset))

    gaz = {
        "adm1_alias": {k: {ak: sorted(av) for ak, av in v.items()} for k, v in country_adm1_alias.items()},
        "adm1_native": {k: {ak: sorted(av) for ak, av in v.items()} for k, v in country_adm1_native.items()},
        "adm2_alias": country_adm2_alias,
        "adm2_native": country_adm2_native,
        "adm3_alias": country_adm3_alias,
        "adm3_native": country_adm3_native,
        "single_gid1": country_single_gid1,
        "abbrev_map": iso_abbrev_map,
        "c0_cents": c0_cents,
    }
    gaz_path.write_text(json.dumps(gaz, ensure_ascii=False), encoding="utf-8")
    return adm1_out, gaz, c0_cents


def fetch_montandon_features(cache_path: Path, stac_url: str, bearer_token: str = "") -> list[dict[str, Any]]:
    """Load cached Montandon STAC JSON or fetch FL/FF impacts (2021-2025) from STAC API."""
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    if not bearer_token:
        raise RuntimeError(f"Montandon cache not found at {cache_path} and MONTANDON_STAC_BEARER_TOKEN is not set.")
    headers = {"Authorization": f"Bearer {bearer_token}", "Content-Type": "application/json"}
    by_id: dict[str, dict[str, Any]] = {}
    for code in ("FL", "FF"):
        q: dict[str, Any] = {
            "collections": ["emdat-impacts", "gdacs-impacts"],
            "filter-lang": "cql2-json",
            "filter": {
                "op": "and",
                "args": [
                    {"op": "a_contains", "args": [{"property": "monty:hazard_codes"}, code]},
                    {"op": ">=", "args": [{"property": "datetime"}, "2021-01-01T00:00:00Z"]},
                    {"op": "<=", "args": [{"property": "datetime"}, "2025-12-31T23:59:59Z"]},
                ],
            },
            "limit": 250,
        }
        while True:
            resp = requests.post(stac_url, json=q, headers=headers, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            feats = payload.get("features") or []
            if not feats:
                break
            for feat in feats:
                by_id[feat["id"]] = feat
            nxt = next((lk for lk in payload.get("links", []) if lk.get("rel") == "next"), None)
            if not nxt or not nxt.get("body"):
                break
            q = nxt["body"]
            time.sleep(0.05)
    feats = list(by_id.values())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(feats), encoding="utf-8")
    return feats


def run_build_catalogs(
    paths: ProjectPaths,
    gadm_gpkg: Path,
    idmc_csv: Path,
    montandon_json: Path,
    idmc_extra_parquet: Path | None = None,
    groundsource_parquet: Path | None = None,
    pop_rule: PopRule = "priority_fallback",
) -> dict[str, pd.DataFrame]:
    """Build GADM Admin-1 layer, Catalog A, and (if Groundsource v2 supplied) Catalog B."""
    sup1 = pd.read_csv(paths.supplementary_data_1_csv, keep_default_na=False)
    sup1_all_185 = set(sup1["ISO3"].str.strip())
    cohort_d_35 = set(sup1[sup1["Cohort_and_Category"].str.startswith("Cohort D")]["ISO3"].str.strip())
    eval_150 = sup1_all_185 - cohort_d_35

    adm1_gdf, gaz, c0_cents = build_gadm_and_gazetteer(gadm_gpkg, paths.work_dir)
    matchers = compile_country_matchers(gaz)

    adm1_geoms = adm1_gdf.geometry.values
    adm1_gids = adm1_gdf["gid_1"].values
    adm1_isos = adm1_gdf["country_iso3"].values
    adm1_areas = adm1_gdf["area_deg2"].values
    adm1_nunits = adm1_gdf["n_units_in_country"].values
    adm1_careas = adm1_gdf["country_area_deg2"].values
    tree = STRtree(adm1_geoms)

    # Part A: IDMC CSV + optional IDMC API/DB Parquet
    df_csv = pd.read_csv(idmc_csv, keep_default_na=False, na_values=[""])
    df_csv = df_csv[
        (df_csv["Year"].between(2021, 2025)) & (df_csv["Hazard Type"].str.contains("Flood", case=False, na=False))
    ].copy()
    df_idmc_csv = pd.DataFrame(
        {
            "event_id": df_csv["Event ID"],
            "iso3": df_csv["ISO3"],
            "event_name": df_csv["Event Name"],
            "start_date": pd.to_datetime(df_csv["Date of Event (start)"]).dt.tz_localize(None).dt.floor("D"),
            "end_date": pd.to_datetime(df_csv["Date of Event (start)"]).dt.tz_localize(None).dt.floor("D"),
            "displacements": pd.to_numeric(df_csv["Disaster Internal Displacements"], errors="coerce").fillna(0.0),
        }
    )
    if idmc_extra_parquet is not None and idmc_extra_parquet.exists():
        df_ext = pd.read_parquet(idmc_extra_parquet)
        df_ext["start_date"] = pd.to_datetime(df_ext["start_date"]).dt.tz_localize(None).dt.floor("D")
        df_ext["end_date"] = (
            pd.to_datetime(df_ext["end_date"]).dt.tz_localize(None).dt.floor("D").fillna(df_ext["start_date"])
        )
        df_ext["displacements"] = pd.to_numeric(df_ext["displacements"], errors="coerce").fillna(0.0)
        seen = set(df_ext["event_id"].dropna().astype(int))
        extra_csv = df_idmc_csv[~df_idmc_csv["event_id"].fillna(-1).astype(int).isin(seen)]
        df_idmc = pd.concat(
            [df_ext[["event_id", "iso3", "event_name", "start_date", "end_date", "displacements"]], extra_csv],
            ignore_index=True,
        )
    else:
        df_idmc = df_idmc_csv
    df_idmc["country_iso3"] = df_idmc["iso3"].astype(str).str.strip().str.upper().replace(ISO_UNIFY)

    idmc_records = []
    by_cc_diag: dict[str, dict[str, int]] = {}
    for _, row in df_idmc.iterrows():
        iso = row["country_iso3"]
        ev_clean = re.sub(r"^[^:]{2,40}:\s*", "", str(row["event_name"] or ""))
        gids_matched = match_text_to_gid1(iso, ev_clean, matchers, allow_single=True)
        by_cc_diag.setdefault(iso, {"idmc_tot": 0, "idmc_sub": 0, "mont_tot": 0, "mont_sub": 0})
        by_cc_diag[iso]["idmc_tot"] += 1
        if not gids_matched:
            continue
        by_cc_diag[iso]["idmc_sub"] += 1
        share = 1.0 / len(gids_matched)
        for g1 in gids_matched:
            idmc_records.append(
                {
                    "gid_1": g1,
                    "country_iso3": iso,
                    "start_date": row["start_date"],
                    "end_date": row["end_date"],
                    "has_idmc": True,
                    "has_montandon": False,
                    "has_groundsource": False,
                    "displacements": float(row["displacements"]) * share,
                    "num_affected": 0.0,
                    "num_killed": 0.0,
                }
            )

    # Part B: Montandon STAC
    stac_url = os.environ.get("MONTANDON_STAC_API_SEARCH", "https://montandon-eoapi-stage.ifrc.org/stac/search")
    feats = fetch_montandon_features(montandon_json, stac_url, os.environ.get("MONTANDON_STAC_BEARER_TOKEN", ""))
    iso_to_adm1_indices: dict[str, list[int]] = {}
    for idx, iso in enumerate(adm1_isos):
        iso_to_adm1_indices.setdefault(iso, []).append(idx)
    iso_to_cname = adm1_gdf.groupby("country_iso3")["COUNTRY"].first().to_dict()

    mont_records = []
    sanitation_audit = []
    for feat in feats:
        props = feat.get("properties", {})
        dt_str = props.get("datetime") or props.get("start_datetime")
        if not dt_str:
            continue
        s_dt = pd.to_datetime(dt_str).tz_localize(None).floor("D")
        e_dt = max(s_dt, pd.to_datetime(props.get("end_datetime") or dt_str).tz_localize(None).floor("D"))
        title = str(props.get("title") or "")
        desc = str(props.get("description") or "")
        m_unit = re.search(r"\[([^\]]+)\]", desc)
        bracket_u = m_unit.group(1).strip().lower() if m_unit else ""

        impacts = props.get("monty:impact_detail") or []
        if isinstance(impacts, dict):
            impacts = [impacts]
        killed, affected = 0.0, 0.0
        for imp in impacts:
            if not isinstance(imp, dict):
                continue
            icat = str(imp.get("category") or "").strip().lower()
            it = str(imp.get("type") or "").strip().lower()
            val = float(imp.get("value") or 0.0)
            valid_human = is_valid_human_montandon_impact(icat, it, bracket_u)
            if "killed" in it or "death" in it:
                if valid_human:
                    killed = max(killed, val)
                elif val > 0:
                    sanitation_audit.append(
                        {
                            "country": ",".join(sorted(set(props.get("monty:country_codes") or []))),
                            "category": icat,
                            "impact_type": it,
                            "unit_field": str(imp.get("unit") or ""),
                            "bracket_unit": bracket_u,
                            "excluded_value": val,
                        }
                    )
            elif "affected" in it or "exposed" in it:
                if valid_human:
                    affected = max(affected, val)
                elif val > 0:
                    sanitation_audit.append(
                        {
                            "country": ",".join(sorted(set(props.get("monty:country_codes") or []))),
                            "category": icat,
                            "impact_type": it,
                            "unit_field": str(imp.get("unit") or ""),
                            "bracket_unit": bracket_u,
                            "excluded_value": val,
                        }
                    )

        geom_obj = None
        if feat.get("geometry"):
            try:
                geom_obj = make_valid(shape(feat["geometry"]))
            except Exception:
                geom_obj = None

        for raw_cc in props.get("monty:country_codes") or []:
            iso = ISO_UNIFY.get(str(raw_cc).strip().upper(), str(raw_cc).strip().upper())
            by_cc_diag.setdefault(iso, {"idmc_tot": 0, "idmc_sub": 0, "mont_tot": 0, "mont_sub": 0})
            by_cc_diag[iso]["mont_tot"] += 1
            c_name = iso_to_cname.get(iso, "")
            title_c = (
                re.sub(rf"^\s*(?:flash\s+)?flood(?:s)?\s+in\s+{re.escape(str(c_name))}\b", "", title, flags=re.I)
                if c_name
                else title
            )
            desc_c = re.sub(rf",\s*{re.escape(str(c_name))}\s*(?=$|[.;])", "", desc, flags=re.I) if c_name else desc
            gids_txt = match_text_to_gid1(iso, f"{title_c} {desc_c}".strip(), matchers, allow_single=True)

            gids_geom: list[str] = []
            if geom_obj is not None and not geom_obj.is_empty and iso in iso_to_adm1_indices:
                c_idxs = iso_to_adm1_indices[iso]
                if geom_obj.geom_type == "Point":
                    is_centroid = False
                    if adm1_nunits[c_idxs[0]] > 1 and not gids_txt:
                        for cx, cy in c0_cents.get(iso, []):
                            if abs(geom_obj.x - cx) < 0.03 and abs(geom_obj.y - cy) < 0.03:
                                is_centroid = True
                                break
                    if not is_centroid:
                        hits = [i for i in tree.query(geom_obj, predicate="intersects") if adm1_isos[i] == iso]
                        if not hits:
                            best_d, best_i = 0.15, None
                            for ci in c_idxs:
                                d = adm1_geoms[ci].distance(geom_obj)
                                if d < best_d:
                                    best_d, best_i = d, ci
                            if best_i is not None:
                                hits = [best_i]
                        gids_geom = [adm1_gids[i] for i in hits]
                else:
                    c_area = adm1_careas[c_idxs[0]]
                    n_units = adm1_nunits[c_idxs[0]]
                    if not (n_units > 1 and c_area > 0 and (geom_obj.area / c_area) >= 0.80):
                        g_area = max(geom_obj.area, 1e-9)
                        try:
                            poly_hits = tree.query(geom_obj, predicate="intersects")
                        except GEOSException:
                            geom_obj = geom_obj.buffer(0)
                            poly_hits = tree.query(geom_obj, predicate="intersects")
                        for i in poly_hits:
                            if adm1_isos[i] != iso:
                                continue
                            try:
                                inter = adm1_geoms[i].intersection(geom_obj).area
                            except GEOSException:
                                inter = adm1_geoms[i].buffer(0).intersection(geom_obj.buffer(0)).area
                            if inter >= min(0.05 * g_area, 0.05 * max(adm1_areas[i], 1e-9)):
                                gids_geom.append(adm1_gids[i])

            gids_final = sorted(set(gids_geom) | set(gids_txt))
            if not gids_final:
                continue
            by_cc_diag[iso]["mont_sub"] += 1
            share = 1.0 / len(gids_final)
            for g1 in gids_final:
                mont_records.append(
                    {
                        "gid_1": g1,
                        "country_iso3": iso,
                        "start_date": s_dt,
                        "end_date": e_dt,
                        "has_idmc": False,
                        "has_montandon": True,
                        "has_groundsource": False,
                        "displacements": 0.0,
                        "num_affected": affected * share,
                        "num_killed": killed * share,
                    }
                )

    paths.results_dir.mkdir(parents=True, exist_ok=True)
    if sanitation_audit:
        raw_san = pd.DataFrame(sanitation_audit)
        san_df = (
            raw_san.groupby(["category", "impact_type", "unit_field", "bracket_unit"], as_index=False)
            .agg(
                n_records=("excluded_value", "size"),
                n_countries=("country", "nunique"),
                total_excluded_value=("excluded_value", "sum"),
                max_excluded_value=("excluded_value", "max"),
            )
            .sort_values("total_excluded_value", ascending=False)
            .reset_index(drop=True)
        )
    else:
        san_df = pd.DataFrame(
            columns=[
                "category",
                "impact_type",
                "unit_field",
                "bracket_unit",
                "n_records",
                "n_countries",
                "total_excluded_value",
                "max_excluded_value",
            ]
        )
    san_df.to_csv(paths.results_dir / "montandon_nonhuman_unit_audit.csv", index=False)
    pd.DataFrame.from_dict(by_cc_diag, orient="index").to_csv(
        paths.results_dir / "catalog_a_subnational_resolution_by_country.csv"
    )

    cat_a_all = cluster_gid1_episodes(pd.DataFrame(idmc_records + mont_records), pop_rule=pop_rule)
    cat_a_all["in_cohort_d_35"] = cat_a_all["country_iso3"].isin(cohort_d_35)
    cat_a_all["in_eval_150"] = cat_a_all["country_iso3"].isin(eval_150)
    cat_a_all["in_audited_185"] = cat_a_all["country_iso3"].isin(sup1_all_185)
    cat_a_all["in_orig_excl_19"] = cat_a_all["country_iso3"].isin(LEGACY_EXCLUDED_19)
    cat_a_eval = cat_a_all[cat_a_all["in_eval_150"]].reset_index(drop=True)

    cat_a_eval.to_parquet(paths.work_dir / "catalog_without_gs_2021_2025.parquet", index=False)
    cat_a_all.to_parquet(paths.work_dir / "catalog_without_gs_all_185_2021_2025.parquet", index=False)

    out = {"cat_a_eval": cat_a_eval, "cat_a_all": cat_a_all}
    if groundsource_parquet is not None and groundsource_parquet.exists():
        from cap_alerting_coverage.pipeline.build_groundsource_catalog_b import map_groundsource_to_adm1_daily

        gs_daily = map_groundsource_to_adm1_daily(groundsource_parquet, adm1_gdf)
        gs_daily.to_parquet(paths.work_dir / "groundsource_gid1_daily_2021_2025.parquet", index=False)
        cat_b_all, _ = merge_strictly_nested_catalog_b(
            cat_a_all, gs_daily, cohort_d_35, eval_150, sup1_all_185, pop_rule=pop_rule
        )
        cat_b_eval = cat_b_all[cat_b_all["in_eval_150"]].reset_index(drop=True)
        cat_b_eval.to_parquet(paths.work_dir / "catalog_with_gs_2021_2025.parquet", index=False)
        cat_b_all.to_parquet(paths.work_dir / "catalog_with_gs_all_185_2021_2025.parquet", index=False)
        out["cat_b_eval"] = cat_b_eval
        out["cat_b_all"] = cat_b_all
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GADM Admin-1 layer and Catalog A / B (Tier 2).")
    parser.add_argument("--gadm-gpkg", required=True, help="Path to gadm_410-levels.gpkg.")
    parser.add_argument("--idmc-csv", required=True, help="Path to IDMC_Events CSV.")
    parser.add_argument("--montandon-json", required=True, help="Path to Montandon STAC JSON cache.")
    parser.add_argument("--idmc-extra-parquet", default=None, help="Optional IDMC API/DB Parquet export.")
    parser.add_argument("--groundsource-parquet", default=None, help="Optional Groundsource v2 GeoParquet.")
    parser.add_argument("--work-dir", default=None, help="Output working directory.")
    parser.add_argument("--results-dir", default=None, help="Output results audit directory.")
    parser.add_argument("--pop-rule", choices=["priority_fallback", "max"], default="priority_fallback")
    args = parser.parse_args()

    paths = ProjectPaths.from_env(work_dir=args.work_dir, results_dir=args.results_dir)
    run_build_catalogs(
        paths=paths,
        gadm_gpkg=Path(args.gadm_gpkg),
        idmc_csv=Path(args.idmc_csv),
        montandon_json=Path(args.montandon_json),
        idmc_extra_parquet=Path(args.idmc_extra_parquet) if args.idmc_extra_parquet else None,
        groundsource_parquet=Path(args.groundsource_parquet) if args.groundsource_parquet else None,
        pop_rule=args.pop_rule,
    )


if __name__ == "__main__":
    main()
