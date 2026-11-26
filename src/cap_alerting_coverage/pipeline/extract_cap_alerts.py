"""Step 02 (Tier 2): Portable CAP alert classification and strict GADM Admin-1 geocoding."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import make_valid
from shapely.errors import GEOSException
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

from cap_alerting_coverage.config import ISO_UNIFY, ProjectPaths
from cap_alerting_coverage.keywords import (
    classify_cap_alert,
    classify_multi_hazard,
    compute_cap_score64,
    map_severity,
)
from cap_alerting_coverage.pipeline.build_catalogs import compile_country_matchers, match_text_to_gid1


def parse_cap_polygon(poly_str: str | None) -> Polygon | None:
    """Parse an OASIS CAP ``'lat,lon lat,lon ...'`` polygon into a valid Shapely ``(lon, lat)`` polygon."""
    if not poly_str or not isinstance(poly_str, str):
        return None
    pts: list[tuple[float, float]] = []
    for tok in poly_str.strip().replace(";", " ").split():
        parts = tok.split(",")
        if len(parts) < 2:
            continue
        try:
            lat = float(parts[0])
            lon = float(parts[1])
        except ValueError:
            continue
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            pts.append((lon, lat))
    if len(pts) < 3:
        return None
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    try:
        g = make_valid(Polygon(pts))
        if g.is_empty:
            return None
        bounds = g.bounds
        if (bounds[2] - bounds[0]) > 180.0:
            return None
        return g
    except Exception:
        return None


class CapAdmin1Geocoder:
    """Strict GADM v4.1 Admin-1 geocoder implementing the 4-stage cascade with zero Admin-0 fallback.

    Cascade order:
      1. Explicit CAP polygon with ``>= 5%`` area overlap (``min(0.05 * poly_area, 0.05 * adm1_area)``),
         rejecting whole-country polygons covering ``>= 80%`` of a multi-unit country.
      2. Explicit lat/lon point (rejecting ``(0, 0)`` and generic country centroids within ``0.03°``),
         with ``<= 0.15°`` coastal nearest fallback.
      3. Gazetteer match on ``area_desc`` + ``headline`` + ``translated_headline``.
      4. Gazetteer match on ``description`` + ``translated_description``.
    """

    def __init__(self, adm1_gdf: gpd.GeoDataFrame, gazetteer: dict[str, Any]) -> None:
        self.adm1_geoms = np.array([make_valid(g) for g in adm1_gdf.geometry.values], dtype=object)
        self.adm1_gids = adm1_gdf["gid_1"].astype(str).to_numpy()
        self.adm1_isos = adm1_gdf["country_iso3"].astype(str).to_numpy()
        self.adm1_areas = adm1_gdf["area_deg2"].to_numpy(dtype=np.float64)
        self.adm1_careas = adm1_gdf["country_area_deg2"].to_numpy(dtype=np.float64)
        self.adm1_nunits = adm1_gdf["n_units_in_country"].to_numpy(dtype=np.int32)
        self.gid_to_iso = dict(zip(self.adm1_gids, self.adm1_isos))
        self.tree = STRtree(self.adm1_geoms)
        self.iso_to_idx: dict[str, list[int]] = {}
        for idx, iso in enumerate(self.adm1_isos):
            self.iso_to_idx.setdefault(iso, []).append(idx)
        iso_to_cname = (
            adm1_gdf.groupby("country_iso3")["COUNTRY"].first().to_dict() if "COUNTRY" in adm1_gdf.columns else {}
        )
        self.iso_cname_re = {
            iso: re.compile(rf"\b{re.escape(str(cname))}\b", re.IGNORECASE)
            for iso, cname in iso_to_cname.items()
            if cname
        }
        self.c0_cents = gazetteer.get("c0_cents", {})
        self.matchers = compile_country_matchers(gazetteer)

    def resolve_polygon(self, iso: str, poly_str: str | None) -> tuple[str, ...]:
        """Map a CAP polygon to GADM Admin-1 units meeting the >=5% overlap rule."""
        c_idxs = self.iso_to_idx.get(iso, ())
        if not poly_str or not c_idxs:
            return ()
        pg = parse_cap_polygon(poly_str)
        if pg is None:
            return ()
        c_area = self.adm1_careas[c_idxs[0]]
        n_units = self.adm1_nunits[c_idxs[0]]
        if n_units > 1 and c_area > 0 and (pg.area / c_area) >= 0.80:
            return ()
        g_area = max(pg.area, 1e-12)
        try:
            cand_idxs = self.tree.query(pg, predicate="intersects")
        except GEOSException:
            pg = pg.buffer(0)
            cand_idxs = self.tree.query(pg, predicate="intersects")
        out: list[str] = []
        for gi in cand_idxs:
            if self.adm1_isos[gi] != iso:
                continue
            try:
                inter = self.adm1_geoms[gi].intersection(pg).area
            except GEOSException:
                inter = self.adm1_geoms[gi].buffer(0).intersection(pg.buffer(0)).area
            thresh = min(0.05 * g_area, 0.05 * max(self.adm1_areas[gi], 1e-12))
            if inter >= thresh:
                out.append(self.adm1_gids[gi])
        if not out and pg.area > 0:
            rp = pg.representative_point()
            for gi in self.tree.query(rp, predicate="intersects"):
                if self.adm1_isos[gi] == iso:
                    out.append(self.adm1_gids[gi])
                    break
        return tuple(sorted(set(out)))

    def resolve_point(self, iso: str, lat: float | None, lon: float | None) -> tuple[str, ...]:
        """Map a CAP point coordinate to GADM Admin-1 (suppressing generic country centroids)."""
        if lat is None or lon is None or not (np.isfinite(lat) and np.isfinite(lon)):
            return ()
        if (abs(lat) < 1e-6 and abs(lon) < 1e-6) or not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return ()
        c_idxs = self.iso_to_idx.get(iso, ())
        if not c_idxs:
            return ()
        if self.adm1_nunits[c_idxs[0]] > 1:
            for cx, cy in self.c0_cents.get(iso, ()):
                if abs(lon - cx) < 0.03 and abs(lat - cy) < 0.03:
                    return ()
        pt = Point(lon, lat)
        hits = [gi for gi in self.tree.query(pt, predicate="intersects") if self.adm1_isos[gi] == iso]
        if not hits:
            best_d, best_i = 0.15, None
            for ci in c_idxs:
                d = self.adm1_geoms[ci].distance(pt)
                if d < best_d:
                    best_d, best_i = d, ci
            if best_i is not None:
                hits = [best_i]
        return tuple(sorted({self.adm1_gids[gi] for gi in hits}))

    def resolve_text(self, iso: str, raw_txt: str, allow_single: bool = True) -> tuple[str, ...]:
        """Match area descriptions or alert headlines against the GADM multilingual gazetteer."""
        cname_re = self.iso_cname_re.get(iso)
        txt = cname_re.sub(" ", raw_txt) if cname_re else raw_txt
        return tuple(match_text_to_gid1(iso, txt, self.matchers, allow_single=allow_single))

    def geocode_alert(self, iso: str, row: dict[str, Any]) -> tuple[tuple[str, ...], str]:
        """Apply the 4-stage geocoding cascade and return ``(matched_gids, method_name)``."""
        poly_s = row.get("polygon") or ""
        if poly_s:
            hits = self.resolve_polygon(iso, poly_s)
            if hits:
                return hits, "poly"
        lat = row.get("latitude")
        lon = row.get("longitude")
        if lat is not None and lon is not None:
            hits = self.resolve_point(iso, round(float(lat), 4), round(float(lon), 4))
            if hits:
                return hits, "point"
        raw_txt = f"{row.get('area_desc') or ''} ; {row.get('headline') or ''} ; {row.get('translated_headline') or ''}"
        hits = self.resolve_text(iso, raw_txt, allow_single=True)
        if hits:
            return hits, "gaz_area"
        d_txt = f"{row.get('description') or ''} {row.get('translated_description') or ''}".strip()
        if d_txt:
            hits = self.resolve_text(iso, d_txt, allow_single=False)
            if hits:
                return hits, "gaz_desc"
        return (), "unmatched"


def aggregate_cap_alerts(
    records: Iterable[dict[str, Any]],
    geocoder: CapAdmin1Geocoder,
) -> dict[str, pd.DataFrame]:
    """Classify, geocode, and aggregate raw CAP alert rows into de-identified daily matrices.

    Raw CAP text, XML, URLs, and polygons are discarded during aggregation; only de-identified
    daily ``gid_1`` / ``country_iso3`` severity/score64/count tables and audit counts are returned.
    """
    gid0_agg: dict[tuple[str, str], list[int]] = {}
    gid1_agg: dict[tuple[str, str, str], list[int]] = {}
    audit_agg: dict[str, list[int]] = {}
    haz_onb_agg: dict[str, list[Any]] = {}

    for rec in records:
        if rec.get("is_from_google"):
            continue
        st_low = str(rec.get("status") or "").strip().lower()
        mt_low = str(rec.get("msg_type") or "").strip().lower()
        if st_low in ("test", "exercise", "draft", "system") or mt_low == "cancel":
            continue
        sent_dt = pd.Timestamp(rec["sent_at"])
        if pd.isna(sent_dt):
            continue
        date_s = sent_dt.strftime("%Y-%m-%d")
        if not ("2020-12-15" <= date_s <= "2025-12-31"):
            continue
        sent_iso = sent_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        cc_raw = str(rec.get("country_code") or "").strip().upper()
        iso = ISO_UNIFY.get(cc_raw, cc_raw)

        is_perm, is_rest, _ = classify_cap_alert(
            event=rec.get("event"),
            headline=rec.get("headline"),
            description=rec.get("description"),
            translated_headline=rec.get("translated_headline"),
            translated_description=rec.get("translated_description"),
        )

        if "2021-01-01" <= date_s <= "2025-12-31" and st_low in ("", "actual") and mt_low in ("", "alert", "update"):
            ho = haz_onb_agg.setdefault(iso, [sent_iso, sent_iso, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ""])
            ho[0] = min(ho[0], sent_iso)
            ho[1] = max(ho[1], sent_iso)
            ho[2 + (int(date_s[:4]) - 2021)] += 1
            hz = classify_multi_hazard(
                rec.get("event"), rec.get("headline"), rec.get("translated_headline"), rec.get("category")
            )
            ho[7] += int(hz["storms"])
            ho[8] += int(hz["cold"])
            ho[9] += int(hz["heat"])
            ho[10] += int(hz["wildfire"])
            ho[11] += int(hz["earthquake"])
            ho[12] += int(hz["volcano"])
            if is_perm:
                ho[13] += 1
                if not ho[15] or sent_iso < ho[15]:
                    ho[15] = sent_iso
            if is_rest:
                ho[14] += 1

        if not is_perm:
            continue

        sev = map_severity(rec.get("severity"))
        score64 = compute_cap_score64(rec.get("severity"), rec.get("certainty"), rec.get("urgency"))
        p_sev, r_sev = sev, (sev if is_rest else 0)
        p_sc, r_sc = score64, (score64 if is_rest else 0)
        np_i, nr_i = 1, (1 if is_rest else 0)

        cur0 = gid0_agg.get((iso, date_s))
        if cur0 is None:
            gid0_agg[(iso, date_s)] = [p_sev, r_sev, p_sc, r_sc, np_i, nr_i]
        else:
            cur0[0] = max(cur0[0], p_sev)
            cur0[1] = max(cur0[1], r_sev)
            cur0[2] = max(cur0[2], p_sc)
            cur0[3] = max(cur0[3], r_sc)
            cur0[4] += np_i
            cur0[5] += nr_i

        gids_hit, meth = geocoder.geocode_alert(iso, rec)
        aud = audit_agg.setdefault(iso, [0, 0, 0, 0, 0, 0, 0, 0])
        aud[0] += 1
        aud[1] += np_i
        aud[2] += nr_i
        meth_map = {"poly": 4, "point": 5, "gaz_area": 6, "gaz_desc": 7}
        if gids_hit:
            aud[3] += 1
            aud[meth_map[meth]] += 1
            for g1 in gids_hit:
                true_iso = geocoder.gid_to_iso.get(g1, iso)
                cur1 = gid1_agg.get((g1, true_iso, date_s))
                if cur1 is None:
                    gid1_agg[(g1, true_iso, date_s)] = [p_sev, r_sev, p_sc, r_sc, np_i, nr_i]
                else:
                    cur1[0] = max(cur1[0], p_sev)
                    cur1[1] = max(cur1[1], r_sev)
                    cur1[2] = max(cur1[2], p_sc)
                    cur1[3] = max(cur1[3], r_sc)
                    cur1[4] += np_i
                    cur1[5] += nr_i

    gid0_df = pd.DataFrame(
        [
            (k[0], k[1], np.int8(v[0]), np.int8(v[1]), np.int16(v[2]), np.int16(v[3]), np.int32(v[4]), np.int32(v[5]))
            for k, v in gid0_agg.items()
        ],
        columns=[
            "country_iso3",
            "alert_date",
            "perm_max_sev",
            "rest_max_sev",
            "perm_max_score64",
            "rest_max_score64",
            "n_perm",
            "n_rest",
        ],
    )
    gid1_df = pd.DataFrame(
        [
            (
                k[0],
                k[1],
                k[2],
                np.int8(v[0]),
                np.int8(v[1]),
                np.int16(v[2]),
                np.int16(v[3]),
                np.int32(v[4]),
                np.int32(v[5]),
            )
            for k, v in gid1_agg.items()
        ],
        columns=[
            "gid_1",
            "country_iso3",
            "alert_date",
            "perm_max_sev",
            "rest_max_sev",
            "perm_max_score64",
            "rest_max_score64",
            "n_perm",
            "n_rest",
        ],
    )
    if not gid0_df.empty:
        gid0_df["alert_date"] = pd.to_datetime(gid0_df["alert_date"])
    if not gid1_df.empty:
        gid1_df["alert_date"] = pd.to_datetime(gid1_df["alert_date"])

    audit_df = pd.DataFrame(
        [
            {
                "country_iso3": iso,
                "total_alerts": v[0],
                "perm_alerts": v[1],
                "rest_alerts": v[2],
                "subnational_matched": v[3],
                "poly_matched": v[4],
                "point_matched": v[5],
                "gaz_area_matched": v[6],
                "gaz_desc_matched": v[7],
                "subnational_pct": round(100.0 * v[3] / max(1, v[0]), 1),
            }
            for iso, v in audit_agg.items()
        ]
    )
    onb_df = pd.DataFrame(
        [
            {
                "country_iso3": iso,
                "first_any_cap_dt": v[0],
                "last_any_cap_dt": v[1],
                "first_flood_perm_dt": v[15],
                "n_any_total": sum(v[2:7]),
                "n_2021": v[2],
                "n_2022": v[3],
                "n_2023": v[4],
                "n_2024": v[5],
                "n_2025": v[6],
                "onboard_year_2021_2025": int(v[0][:4]) if v[0] else 0,
            }
            for iso, v in sorted(haz_onb_agg.items())
        ]
    )
    haz_df = pd.DataFrame(
        [
            {
                "country_iso3": iso,
                "n_any_total": sum(v[2:7]),
                "n_storms": v[7],
                "n_cold": v[8],
                "n_heat": v[9],
                "n_wildfire": v[10],
                "n_earthquake": v[11],
                "n_volcano": v[12],
                "n_flood_perm": v[13],
                "n_flood_rest": v[14],
            }
            for iso, v in sorted(haz_onb_agg.items())
        ]
    )
    return {
        "cap_gid1_daily": gid1_df,
        "cap_gid0_daily": gid0_df,
        "geocoding_audit": audit_df,
        "onboarding": onb_df,
        "hazard_counts": haz_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract de-identified CAP Admin-1 matrices from SQL or Parquet (Tier 2)."
    )
    parser.add_argument("--cap-parquet", default=None, help="Optional input Parquet of raw CAP alert rows.")
    parser.add_argument("--db-url", default=None, help="Optional SQLAlchemy DB URL (or set CAP_GAP_DB_URL).")
    parser.add_argument("--work-dir", default=None, help="Working directory with gadm41_adm1.parquet & gazetteer.")
    parser.add_argument("--results-dir", default=None, help="Results directory for onboarding/hazard/geocoding CSVs.")
    args = parser.parse_args()

    paths = ProjectPaths.from_env(work_dir=args.work_dir, results_dir=args.results_dir)
    adm1 = gpd.read_parquet(paths.work_dir / "gadm41_adm1.parquet")
    gaz = json.loads((paths.work_dir / "gadm41_gazetteer.json").read_text(encoding="utf-8"))
    geocoder = CapAdmin1Geocoder(adm1, gaz)

    if args.cap_parquet:
        df_in = pd.read_parquet(args.cap_parquet)
        records = df_in.to_dict("records")
    else:
        import sqlalchemy

        from cap_alerting_coverage.ingestion.db import get_sqlalchemy_engine

        engine = get_sqlalchemy_engine(args.db_url)
        query = sqlalchemy.text("""
            SELECT country_code, sent_at, event, severity, headline, description, area_desc,
                   polygon, latitude, longitude, translated_headline, translated_description,
                   certainty, urgency, category, status, msg_type, is_from_google
            FROM cap_alerts
            WHERE sent_at >= '2020-12-15' AND sent_at < '2026-01-01'
        """)
        with engine.connect() as conn:
            records = [dict(r._mapping) for r in conn.execute(query)]

    outputs = aggregate_cap_alerts(records, geocoder)
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    outputs["cap_gid1_daily"].to_parquet(paths.work_dir / "cap_gid1_daily_2020_2025.parquet", index=False)
    outputs["cap_gid0_daily"].to_parquet(paths.work_dir / "cap_gid0_daily_2020_2025.parquet", index=False)
    outputs["geocoding_audit"].to_csv(paths.results_dir / "cap_geocoding_audit_by_country.csv", index=False)
    outputs["onboarding"].to_csv(paths.results_dir / "cap_country_onboarding_2021_2025.csv", index=False)
    outputs["hazard_counts"].to_csv(paths.results_dir / "cap_country_hazard_counts_2021_2025.csv", index=False)
    print(f"Saved {len(outputs['cap_gid1_daily']):,} de-identified CAP GID_1-days to {paths.work_dir}")


if __name__ == "__main__":
    main()
