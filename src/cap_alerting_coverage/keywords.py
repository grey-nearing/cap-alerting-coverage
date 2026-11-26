"""Multilingual CAP hazard classification filters and OASIS CAP v1.2 priority scoring."""

from __future__ import annotations

import re

# Lowercase patterns compiled without re.IGNORECASE for deterministic fast matching on lowercased inputs.
EXCL_REGEX_STR = (
    r"snow|\bice\b|heat|fire|temperature|frost|freeze|cold|drought|\bdry\b|dust|\bfog\b|gale|breeze|wind|"
    r"avalanche|earthquake|seismic|forest|neige|glace|\bgel\b|froid|chaleur|canicule|sécheresse|\bsec\b|"
    r"poussière|brouillard|\bfeu\b|incendie|séisme|avalancha|nieve|hielo|helada|calor|sequía|seco|polvo|"
    r"niebla|fuego|sismo|terremoto|deslizamiento|снег|лед|мороз|заморозки|холод|жара|засуха|сухо|пыль|"
    r"туман|пожар|землетрясение|лавина|оползень|雪|冰|霜|冻|冷|热|热浪|干旱|干燥|尘|雾|火|火灾|地震|"
    r"海啸|雪崩|滑坡|泥石流|ثلج|جليد|صقيع|برد|حر|موجة|جفاف|جاف|غبار|ضباب|حريق|زلزال|انهيار|"
    r"برف|یخبندان|سرما|گرما|خشکسالی|گردوخاک|مه|آتش|زلزله|"
    r"lumi|jää|pakkanen|kylmä|kuumuus|helleaalto|kuivuus|pöly|sumu|tulipalo|maanjäristys|lumivyöry|"
    r"maanvyörymä|snø|\bis\b|kulde|varme|varmebølge|tørke|støv|tåke|brann|skogsbrann|jordskjelv|"
    r"snøskred|jordskred|snijeg|led|mraz|hladno|vrućina|toplota|suša|prašina|magla|vjetar|požar|"
    r"zemljotres|lavina|klizište|sneh|ľad|mráz|chlad|teplo|horúčava|sucho|prach|hmla|vietor|požiar|"
    r"zemetrasenie|lavína|zosuv|neve|gelo|geada|frio|calor|seca|poeira|nevoeiro|fogo|incêndio|"
    r"desmoronamento|salju|\bes\b|embun beku|dingin|panas|kemarau|kering|debu|kabut|kebakaran|gempa|"
    r"tsunami|longsor|niebe|yelo|lamig|init|tuyot|tuyo|alikabok|hamog|sunog|lindol|luho"
)

PERM_REGEX_STR = (
    r"flood|rain|storm|cyclone|typhoon|hurricane|surge|water|inundat|overflow|\bwet\b|precipitation|"
    r"thunderstorm|hydrolog|flash|rising|\bdam\b|inondation|pluie|précipitation|orage|ouragan|tempête|"
    r"\bcrue\b|déluge|lluvia|precipitación|tormenta|ciclón|huracán|crecida|diluvio|наводнение|дождь|"
    r"осадки|шторм|циклон|ураган|паводок|ливень|затопление|половодье|洪水|雨|暴雨|降水|降雨|台风|"
    r"飓风|淹没|内涝|山洪|فيضان|سيول|مطر|أمطار|عاصفة|إعصار|طوفان|سیل|سیلاب|آبگرفتگی|رواناب|"
    r"بارش شدید|بارش|طوفان|tulva|sade|rankkasade|sademäärä|myrsky|pyörremyrsky|\bflom\b|regn|nedbør|"
    r"orkan|inundação|chuva|tempestade|cheia|alagamento|enxurrada|furacão|banjir|waspada|siklon|topan|"
    r"badai|genangan|\bbaha\b|ulan|bagyo|monsoon|poplav|bujic|kiša|pljusak|padavine|nevrijeme|"
    r"povodeň|povodň|záplav|zrážky|lejú|prší|víchrica|zaplavenie|hochwasser|überschwemmung|sturzflut|"
    r"unwetter|starkregen|regen|alluvione|esondazione|pioggia|nubifragio|taşkın|yağmur|sağanak|\bsel\b|"
    r"น้ำท่วม|ฝนตกหนัก|ฝน|พายุ|\blũ\b|ngập|mưa|bão|홍수|호우|태풍|침수|폭우|बाढ़|भारी बारिश|बारिश|तूफान"
)

REST_REGEX_STR = (
    r"flood|inundat|overflow|hydrolog|river level|rising water|inondation|\bcrue\b|crue torrentielle|"
    r"crecida|desbordamiento|inundación|inundação|enxurrada|alagamento|\bcheia\b|transbordamento|"
    r"наводнение|паводок|затопление|половодье|подтопление|洪水|内涝|山洪|فيضان|سيول|غمر|"
    r"سیل|سیلاب|آبگرفتگی|رواناب|طغیان|tulva|\bflom\b|oversvømmelse|översvämning|"
    r"banjir|genangan|\bbaha\b|pagbaha|poplav|bujic|povodeň|povodň|záplav|zaplavenie|"
    r"hochwasser|überschwemmung|sturzflut|alluvione|esondazione|inondazione|"
    r"taşkın|\bsel\b|su baskını|\blũ\b|ngập lụt|ngập úng|น้ำท่วม|น้ำป่า|น้ำล้นตลิ่ง|"
    r"홍수|침수|범람|बाढ़|जलभराव"
)

EXCL_RE = re.compile(EXCL_REGEX_STR)
PERM_RE = re.compile(PERM_REGEX_STR)
REST_RE = re.compile(REST_REGEX_STR)

HAZ_STORMS_RE = re.compile(
    r"storm|cyclone|typhoon|hurricane|thunderstorm|gale|squall|tornado|wind|tempête|ouragan|vent|"
    r"tormenta|huracán|sturm|unwetter|шторм|циклон|ураган|عاصفة|طوفان|تندباد|台风|飓风|暴风|大风|"
    r"myrsky|tempestade|furacão|badai|siklon|topan|fırtına|พายุ|bão|태풍|폭풍|강풍|तूफान|poplav.*storm|nevrijeme"
)
HAZ_COLD_RE = re.compile(
    r"cold|frost|freeze|freezing|snow|ice|icy|blizzard|avalanche|neige|\bgel\b|froid|nieve|hielo|helada|frío|"
    r"schnee|kälte|снег|лед|мороз|холод|ثلج|جليد|صقيع|برد|برف|سرما|雪|冰|霜|冻|寒潮|lumi|pakkanen|"
    r"snø|kulde|neve|gelo|mráz|snijeg|mraz|sneh|buz|soğuk|한파|대설|schneefall"
)
HAZ_HEAT_RE = re.compile(
    r"heat|high temp|extreme temp|chaleur|canicule|calor|ola de calor|hitze|wärme|жара|"
    r"\bحر\b|موجة حارة|گرما|热浪|高温|helle|varme|caldo|varmebølge|vrućina|horúčava|vedro|sıcak|폭염|लूं"
)
HAZ_WILDFIRE_RE = re.compile(
    r"\b(?:wildfire|bushfire|forest\s*fire|brush\s*fire|fire\s*warning|fire\s*weather|fire\s*danger|grass\s*fire)s?\b|"
    r"incendie|feu de forêt|incendio forestal|waldbrand|пожар|حريق|آتش.?سوزی|山火|森林火灾|maastopalo|"
    r"skogbrann|incêndio|požar|yangın|ไฟป่า|cháy rừng|산불|दावानल"
)
HAZ_QUAKE_RE = re.compile(
    r"earthquake|seismic|tsunami|séisme|tremblement|terremoto|sismo|erdbeben|землетрясение|"
    r"زلزال|زلزله|地震|海啸|maanjäristys|jordskjelv|terramoto|potres|zemetrasenie|deprem|แผ่นดินไหว|động đất|지진|भूकंप"
)
HAZ_VOLCANO_RE = re.compile(
    r"volcan|ashfall|volcanic|eruption|volcán|vulkan|вулкан|بركان|آتشفشان|火山|tulivuori|vulcão|vulcano|yanardağ|화산|ज्वालामुखी"
)


def map_severity(sev_val: str | None) -> int:
    """Map OASIS CAP v1.2 <severity> string to ordinal rank 1..4."""
    s = (sev_val or "").strip().lower()
    if s == "extreme":
        return 4
    if s == "severe":
        return 3
    if s == "moderate":
        return 2
    return 1


def map_certainty(cert_val: str | None) -> int:
    """Map OASIS CAP v1.2 <certainty> string to ordinal rank 1..4."""
    s = (cert_val or "").strip().lower()
    if s == "observed":
        return 4
    if s == "likely":
        return 3
    if s == "possible":
        return 2
    return 1


def map_urgency(urg_val: str | None) -> int:
    """Map OASIS CAP v1.2 <urgency> string to ordinal rank 1..4."""
    s = (urg_val or "").strip().lower()
    if s == "immediate":
        return 4
    if s == "expected":
        return 3
    if s == "future":
        return 2
    return 1


def compute_cap_score64(severity: str | int | None, certainty: str | int | None, urgency: str | int | None) -> int:
    """Compute the 64-rank lexicographic OASIS CAP v1.2 priority score in 1..64.

    Severity is placed first, certainty second, urgency third:
    ``score64 = 16*(sev - 1) + 4*(cert - 1) + (urg - 1) + 1``.
    Boundary cutoffs 1, 17, 33, 49 correspond exactly to All, >=Moderate, >=Severe, and Extreme.
    """
    sev = severity if isinstance(severity, int) else map_severity(severity)
    cert = certainty if isinstance(certainty, int) else map_certainty(certainty)
    urg = urgency if isinstance(urgency, int) else map_urgency(urgency)
    if not (1 <= sev <= 4 and 1 <= cert <= 4 and 1 <= urg <= 4):
        raise ValueError(f"Expected ordinal ranks in 1..4, got sev={sev}, cert={cert}, urg={urg}.")
    return 16 * (sev - 1) + 4 * (cert - 1) + (urg - 1) + 1


def classify_cap_alert(
    event: str | None = None,
    headline: str | None = None,
    description: str | None = None,
    translated_headline: str | None = None,
    translated_description: str | None = None,
) -> tuple[bool, bool, bool]:
    """Classify a CAP alert into (is_broad_flood_storm, is_explicit_flood_only, is_excluded).

    Preserves the exact classification logic used in the benchmark extraction pipeline:
      1. ``is_excluded`` triggers when the ``event`` field matches a non-hydrological exclusion stem.
      2. ``is_explicit_flood_only`` requires ``not is_excluded`` and an explicit flood stem in
         ``event``, ``headline``, or ``translated_headline``.
      3. ``is_broad_flood_storm`` is True whenever ``is_explicit_flood_only`` is True, or when
         ``not is_excluded`` and a broad flood/storm/heavy-rain stem matches ``event``, ``headline``,
         ``translated_headline``, ``description``, or ``translated_description``.
    """
    ev_low = (event or "").lower()
    hl_low = (headline or "").lower()
    th_low = (translated_headline or "").lower()

    is_excl = bool(ev_low and EXCL_RE.search(ev_low))
    is_rest = (not is_excl) and bool(
        (ev_low and REST_RE.search(ev_low))
        or (hl_low and REST_RE.search(hl_low))
        or (th_low and REST_RE.search(th_low))
    )
    if is_rest:
        is_perm = True
    elif is_excl:
        is_perm = False
    else:
        is_perm = bool(
            (ev_low and PERM_RE.search(ev_low))
            or (hl_low and PERM_RE.search(hl_low))
            or (th_low and PERM_RE.search(th_low))
        )
        if not is_perm:
            d_low = (description or "").lower()
            td_low = (translated_description or "").lower()
            is_perm = bool((d_low and PERM_RE.search(d_low)) or (td_low and PERM_RE.search(td_low)))
    return is_perm, is_rest, is_excl


def classify_multi_hazard(
    event: str | None = None,
    headline: str | None = None,
    translated_headline: str | None = None,
    category: str | None = None,
) -> dict[str, bool]:
    """Classify non-exclusive multi-hazard CAP indicators for Figure 1b."""
    haz_s = f"{(event or '').lower()} {(headline or '').lower()} {(translated_headline or '').lower()}"
    cat_low = (category or "").strip().lower()
    return {
        "storms": bool(HAZ_STORMS_RE.search(haz_s)),
        "cold": bool(HAZ_COLD_RE.search(haz_s)),
        "heat": bool(HAZ_HEAT_RE.search(haz_s)),
        "wildfire": bool(HAZ_WILDFIRE_RE.search(haz_s) or cat_low == "fire"),
        "earthquake": bool(HAZ_QUAKE_RE.search(haz_s)),
        "volcano": bool(HAZ_VOLCANO_RE.search(haz_s)),
    }
