import json
import os
import re
import uuid
from pathlib import Path
from urllib.parse import urlparse

# Complete ISO-3166-1 numeric -> (ISO2, ISO3) mapping for WMO OIDs (2.49.0.[01].<country_num>.<agency_seq>)
ISO_NUM_MAP = {
    "4": ("AF", "AFG"),
    "8": ("AL", "ALB"),
    "12": ("DZ", "DZA"),
    "16": ("AS", "ASM"),
    "20": ("AD", "AND"),
    "24": ("AO", "AGO"),
    "28": ("AG", "ATG"),
    "32": ("AR", "ARG"),
    "36": ("AU", "AUS"),
    "40": ("AT", "AUT"),
    "44": ("BS", "BHS"),
    "48": ("BH", "BHR"),
    "50": ("BD", "BGD"),
    "51": ("AM", "ARM"),
    "52": ("BB", "BRB"),
    "56": ("BE", "BEL"),
    "64": ("BT", "BTN"),
    "68": ("BO", "BOL"),
    "70": ("BA", "BIH"),
    "72": ("BW", "BWA"),
    "76": ("BR", "BRA"),
    "84": ("BZ", "BLZ"),
    "90": ("SB", "SLB"),
    "96": ("BN", "BRN"),
    "100": ("BG", "BGR"),
    "104": ("MM", "MMR"),
    "108": ("BI", "BDI"),
    "112": ("BY", "BLR"),
    "116": ("KH", "KHM"),
    "120": ("CM", "CMR"),
    "124": ("CA", "CAN"),
    "132": ("CV", "CPV"),
    "140": ("CF", "CAF"),
    "144": ("LK", "LKA"),
    "148": ("TD", "TCD"),
    "152": ("CL", "CHL"),
    "156": ("CN", "CHN"),
    "158": ("TW", "TWN"),
    "170": ("CO", "COL"),
    "174": ("KM", "COM"),
    "178": ("CG", "COG"),
    "180": ("CD", "COD"),
    "188": ("CR", "CRI"),
    "191": ("HR", "HRV"),
    "192": ("CU", "CUB"),
    "196": ("CY", "CYP"),
    "203": ("CZ", "CZE"),
    "204": ("BJ", "BEN"),
    "208": ("DK", "DNK"),
    "212": ("DM", "DMA"),
    "214": ("DO", "DOM"),
    "218": ("EC", "ECU"),
    "222": ("SV", "SLV"),
    "226": ("GQ", "GNQ"),
    "231": ("ET", "ETH"),
    "232": ("ER", "ERI"),
    "233": ("EE", "EST"),
    "242": ("FJ", "FJI"),
    "246": ("FI", "FIN"),
    "250": ("FR", "FRA"),
    "262": ("DJ", "DJI"),
    "266": ("GA", "GAB"),
    "268": ("GE", "GEO"),
    "270": ("GM", "GMB"),
    "275": ("PS", "PSE"),
    "276": ("DE", "DEU"),
    "288": ("GH", "GHA"),
    "300": ("GR", "GRC"),
    "308": ("GD", "GRD"),
    "316": ("GU", "GUM"),
    "320": ("GT", "GTM"),
    "324": ("GN", "GIN"),
    "328": ("GY", "GUY"),
    "332": ("HT", "HTI"),
    "340": ("HN", "HND"),
    "344": ("HK", "HKG"),
    "348": ("HU", "HUN"),
    "352": ("IS", "ISL"),
    "356": ("IN", "IND"),
    "360": ("ID", "IDN"),
    "364": ("IR", "IRN"),
    "368": ("IQ", "IRQ"),
    "372": ("IE", "IRL"),
    "376": ("IL", "ISR"),
    "380": ("IT", "ITA"),
    "384": ("CI", "CIV"),
    "388": ("JM", "JAM"),
    "392": ("JP", "JPN"),
    "398": ("KZ", "KAZ"),
    "400": ("JO", "JOR"),
    "404": ("KE", "KEN"),
    "410": ("KR", "KOR"),
    "414": ("KW", "KWT"),
    "417": ("KG", "KGZ"),
    "418": ("LA", "LAO"),
    "422": ("LB", "LBN"),
    "426": ("LS", "LSO"),
    "428": ("LV", "LVA"),
    "430": ("LR", "LBR"),
    "434": ("LY", "LBY"),
    "440": ("LT", "LTU"),
    "442": ("LU", "LUX"),
    "446": ("MO", "MAC"),
    "450": ("MG", "MDG"),
    "454": ("MW", "MWI"),
    "458": ("MY", "MYS"),
    "462": ("MV", "MDV"),
    "466": ("ML", "MLI"),
    "470": ("MT", "MLT"),
    "478": ("MR", "MRT"),
    "480": ("MU", "MUS"),
    "484": ("MX", "MEX"),
    "496": ("MN", "MNG"),
    "498": ("MD", "MDA"),
    "499": ("ME", "MNE"),
    "504": ("MA", "MAR"),
    "508": ("MZ", "MOZ"),
    "512": ("OM", "OMN"),
    "516": ("NA", "NAM"),
    "524": ("NP", "NPL"),
    "528": ("NL", "NLD"),
    "530": ("CW", "CUW"),
    "531": ("CW", "CUW"),
    "533": ("AW", "ABW"),
    "540": ("NC", "NCL"),
    "548": ("VU", "VUT"),
    "554": ("NZ", "NZL"),
    "558": ("NI", "NIC"),
    "562": ("NE", "NER"),
    "566": ("NG", "NGA"),
    "570": ("NU", "NIU"),
    "578": ("NO", "NOR"),
    "580": ("MP", "MNP"),
    "583": ("FM", "FSM"),
    "584": ("MH", "MHL"),
    "585": ("PW", "PLW"),
    "586": ("PK", "PAK"),
    "591": ("PA", "PAN"),
    "598": ("PG", "PNG"),
    "600": ("PY", "PRY"),
    "604": ("PE", "PER"),
    "608": ("PH", "PHL"),
    "616": ("PL", "POL"),
    "620": ("PT", "PRT"),
    "624": ("GW", "GNB"),
    "626": ("TL", "TLS"),
    "630": ("PR", "PRI"),
    "634": ("QA", "QAT"),
    "642": ("RO", "ROU"),
    "643": ("RU", "RUS"),
    "646": ("RW", "RWA"),
    "659": ("KN", "KNA"),
    "660": ("AI", "AIA"),
    "662": ("LC", "LCA"),
    "670": ("VC", "VCT"),
    "674": ("SM", "SMR"),
    "678": ("ST", "STP"),
    "682": ("SA", "SAU"),
    "686": ("SN", "SEN"),
    "688": ("RS", "SRB"),
    "690": ("SC", "SYC"),
    "694": ("SL", "SLE"),
    "702": ("SG", "SGP"),
    "703": ("SK", "SVK"),
    "704": ("VN", "VNM"),
    "705": ("SI", "SVN"),
    "706": ("SO", "SOM"),
    "710": ("ZA", "ZAF"),
    "716": ("ZW", "ZWE"),
    "724": ("ES", "ESP"),
    "728": ("SS", "SSD"),
    "729": ("SD", "SDN"),
    "740": ("SR", "SUR"),
    "748": ("SZ", "SWZ"),
    "752": ("SE", "SWE"),
    "756": ("CH", "CHE"),
    "760": ("SY", "SYR"),
    "762": ("TJ", "TJK"),
    "764": ("TH", "THA"),
    "768": ("TG", "TGO"),
    "776": ("TO", "TON"),
    "780": ("TT", "TTO"),
    "784": ("AE", "ARE"),
    "788": ("TN", "TUN"),
    "792": ("TR", "TUR"),
    "795": ("TM", "TKM"),
    "796": ("TC", "TCA"),
    "798": ("TV", "TUV"),
    "800": ("UG", "UGA"),
    "804": ("UA", "UKR"),
    "807": ("MK", "MKD"),
    "818": ("EG", "EGY"),
    "826": ("GB", "GBR"),
    "834": ("TZ", "TZA"),
    "840": ("US", "USA"),
    "850": ("VI", "VIR"),
    "854": ("BF", "BFA"),
    "858": ("UY", "URY"),
    "860": ("UZ", "UZB"),
    "862": ("VE", "VEN"),
    "882": ("WS", "WSM"),
    "887": ("YE", "YEM"),
    "894": ("ZM", "ZMB"),
}

ISO2_TO_ISO3 = {v[0]: v[1] for v in ISO_NUM_MAP.values()}
ISO2_TO_ISO3.update({"UK": "GBR", "XK": "XKX", "TW": "TWN", "MO": "MAC", "HK": "HKG"})
ISO3_TO_ISO2 = {v: k for k, v in ISO2_TO_ISO3.items()}

# High-precision sender/domain -> (source_feed_id, agency_id, country_code)
EXPLICIT_AGENCY_PATTERNS = [
    ("mecom.ru", ("ru-hydromet-ru", "RU_HYDROMET", "RUS")),
    ("meteoinfo.ru", ("ru-hydromet-ru", "RU_HYDROMET", "RUS")),
    ("ntwc", ("us-noaa-ntwc-en", "US_NOAA_NTWC", "USA")),
    ("ptwc", ("us-noaa-ptwc-en", "US_NOAA_PTWC", "USA")),
    ("noaa.gov", ("us-noaa-nws-en", "US_NOAA_NWS", "USA")),
    ("weather.gov", ("us-noaa-nws-en", "US_NOAA_NWS", "USA")),
    ("airnow.gov", ("us-epa-aq-en", "US_EPA_AQ", "USA")),
    ("usgs.gov", ("us-usgs-volcano-en", "US_USGS", "USA")),
    ("ipaws", ("us-fema-ipaws-en", "US_FEMA_IPAWS", "USA")),
    ("fema", ("us-fema-ipaws-en", "US_FEMA_IPAWS", "USA")),
    ("bmkg.go.id", ("id-bmkg-id", "ID_BMKG", "IDN")),
    ("dwd.de", ("de-dwd-de", "DE_DWD", "DEU")),
    ("meteo.kz", ("kz-hydromet-ru", "KZ_HYDROMET", "KAZ")),
    ("nrcan-rncan.gc.ca", ("ca-nrcan-en", "CA_NRCAN", "CAN")),
    ("canada.ca", ("ca-eccc-en", "CA_ECCC", "CAN")),
    ("zamg.ac.at", ("at-zamg-de", "AT_ZAMG", "AUT")),
    ("geosphere.at", ("at-zamg-de", "AT_ZAMG", "AUT")),
    ("aemet.es", ("es-aemet-es", "ES_AEMET", "ESP")),
    ("imgw.pl", ("pl-imgw-pl", "PL_IMGW", "POL")),
    ("pogodynka.pl", ("pl-imgw-pl", "PL_IMGW", "POL")),
    ("meteo.dz", ("dz-meteo-en", "DZ_METEO", "DZA")),
    ("office national de la météorologie", ("dz-meteo-en", "DZ_METEO", "DZA")),
    ("gov.uk/environment-agency", ("gb-ea-en", "GB_EA", "GBR")),
    ("metoffice.gov.uk", ("gb-metoffice-en", "GB_METOFFICE", "GBR")),
    ("shmu.sk", ("sk-shmu-sk", "SK_SHMU", "SVK")),
    ("meteoswiss.ch", ("ch-meteoswiss-de", "CH_METEOSWISS", "CHE")),
    ("nve.no", ("no-nve-no", "NO_NVE", "NOR")),
    ("met.no", ("no-met-en", "NO_MET", "NOR")),
    ("fmi.fi", ("fi-fmi-fi", "FI_FMI", "FIN")),
    ("2.49.0.0.246.0", ("fi-fmi-fi", "FI_FMI", "FIN")),
    ("rijkswaterstaat.nl", ("nl-rws-nl", "NL_RWS", "NLD")),
    ("knmi.nl", ("nl-knmi-nl", "NL_KNMI", "NLD")),
    ("aeronautica.difesa.it", ("it-am-it", "IT_AM", "ITA")),
    ("protezionecivile.it", ("it-dpc-it", "IT_DPC", "ITA")),
    ("meteo.fr", ("fr-meteofrance-fr", "FR_METEOFRANCE", "FRA")),
    ("meteofrance", ("fr-meteofrance-fr", "FR_METEOFRANCE", "FRA")),
    ("hidmet.gov.rs", ("rs-rhmz-sr", "RS_RHMZ", "SRB")),
    ("hnms.gr", ("gr-hnms-el", "GR_HNMS", "GRC")),
    ("meteo.kg", ("kg-hydromet-ru", "KG_HYDROMET", "KGZ")),
    ("meteo.hr", ("hr-dhmz-hr", "HR_DHMZ", "HRV")),
    ("smn.gov.ar", ("ar-smn-es", "AR_SMN", "ARG")),
    ("smn.gob.ar", ("ar-smn-es", "AR_SMN", "ARG")),
    ("smn.ar", ("ar-smn-es", "AR_SMN", "ARG")),
    ("water.gov.tw", ("tw-wra-zh", "TW_WRA", "TWN")),
    ("freeway.gov.tw", ("tw-freeway-zh", "TW_FREEWAY", "TWN")),
    ("cwb.gov.tw", ("tw-cwa-zh", "TW_CWA", "TWN")),
    ("cwa.gov.tw", ("tw-cwa-zh", "TW_CWA", "TWN")),
    ("thb.gov.tw", ("tw-thb-zh", "TW_THB", "TWN")),
    (".gov.tw", ("tw-ncdr-zh", "TW_NCDR", "TWN")),
    ("ncdr.nat.gov.tw", ("tw-ncdr-zh", "TW_NCDR", "TWN")),
    ("hmc.by", ("by-belhydromet-ru", "BY_BELHYDROMET", "BLR")),
    ("chmi.cz", ("cz-chmi-cs", "CZ_CHMI", "CZE")),
    ("ideam.gov.co", ("co-ideam-es", "CO_IDEAM", "COL")),
    ("ungrd", ("co-ungrd-es", "CO_UNGRD", "COL")),
    ("kmi-irm", ("be-rmi-nl", "BE_RMI", "BEL")),
    ("meteo.be", ("be-rmi-nl", "BE_RMI", "BEL")),
    ("ipma.pt", ("pt-ipma-pt", "PT_IPMA", "PRT")),
    ("gdacs.org", ("int-gdacs-en", "INT_GDACS", "Other")),
    ("inmet.gov.br", ("br-inmet-pt", "BR_INMET", "BRA")),
    ("bom.gov.au", ("au-bom-en", "AU_BOM", "AUS")),
    ("rfs.nsw.gov.au", ("au-nsw-rfs", "AU_NSW_RFS", "AUS")),
    ("esa.act.gov.au", ("au-act-esa", "AU_ACT_ESA", "AUS")),
    ("qfes.qld.gov.au", ("au-qld-qfes", "AU_QLD_QFES", "AUS")),
    ("cfs.sa.gov.au", ("au-sa-cfs", "AU_SA_CFS", "AUS")),
    ("fire.tas.gov.au", ("au-tas-tfs", "AU_TAS_TFS", "AUS")),
    ("dfes.wa.gov.au", ("au-wa-dfes", "AU_WA_DFES", "AUS")),
    ("emergency.vic.gov.au", ("au-vic-emv", "AU_VIC_EMV", "AUS")),
    ("envir.ee", ("ee-kaur-et", "EE_KAUR", "EST")),
    ("emhi.ee", ("ee-kaur-et", "EE_KAUR", "EST")),
    ("ilmateenistus.ee", ("ee-kaur-et", "EE_KAUR", "EST")),
    ("metservice.com", ("nz-metservice-en", "NZ_METSERVICE", "NZL")),
    ("metservice.co.nz", ("nz-metservice-en", "NZ_METSERVICE", "NZL")),
    ("arso.gov.si", ("si-arso-sl", "SI_ARSO", "SVN")),
    ("met.ie", ("ie-met-en", "IE_MET", "IRL")),
    ("kma.go.kr", ("kr-kma-weather-en", "KR_KMA", "KOR")),
    ("weather.go.kr", ("kr-kma-weather-en", "KR_KMA", "KOR")),
    ("korea.kr", ("kr-kma-weather-en", "KR_KMA", "KOR")),
    ("smhi.se", ("se-smhi-sv", "SE_SMHI", "SWE")),
    ("meteo.lt", ("lt-lhms-xx", "LT_LHMS", "LTU")),
    ("met.hu", ("hu-omsz-hu", "HU_OMSZ", "HUN")),
    ("meteo.uz", ("uz-hydromet-ru", "UZ_HYDROMET", "UZB")),
    ("hko.gov.hk", ("hk-hko-en", "HK_HKO", "HKG")),
    ("meteo.bg", ("bg-nimh-bg", "BG_NIMH", "BGR")),
    ("pagasa", ("ph-pagasa-en", "PH_PAGASA", "PHL")),
    ("dmi.dk", ("dk-dmi-da", "DK_DMI", "DNK")),
    ("conagua.gob.mx", ("mx-smn-es", "MX_SMN", "MEX")),
    ("smn.cna.gob.mx", ("mx-smn-es", "MX_SMN", "MEX")),
    ("rhmzrs.com", ("ba-rhmzrs-sr", "BA_RHMZRS", "BIH")),
    ("fhmzbih.gov.ba", ("ba-fhmz-bs", "BA_FHMZ", "BIH")),
    ("saws", ("za-saws-en", "ZA_SAWS", "ZAF")),
    ("weathersa.co.za", ("za-saws-en", "ZA_SAWS", "ZAF")),
    ("ims.gov.il", ("il-ims-en", "IL_IMS", "ISR")),
    ("meteo.gov.mk", ("mk-uhmr-mk", "MK_UHMR", "MKD")),
    ("meteo.gov.ua", ("ua-uhmc-uk", "UA_UHMC", "UKR")),
    ("meteochile.cl", ("cl-dmc-es", "CL_DMC", "CHL")),
    ("meteochile.gob.cl", ("cl-dmc-es", "CL_DMC", "CHL")),
    ("inumet.gub.uy", ("uy-inumet-es", "UY_INUMET", "URY")),
    ("ncm.gov.sa", ("sa-ncm-en", "SA_NCM", "SAU")),
    ("lvgmc.lv", ("lv-lvgmc-lv", "LV_LVGMC", "LVA")),
    ("meteoromania.ro", ("ro-anm-ro", "RO_ANM", "ROU")),
    ("vedur.is", ("is-imo-en", "IS_IMO", "ISL")),
    ("indomet.gov.do", ("do-indomet-es", "DO_INDOMET", "DOM")),
    ("onamet.gov.do", ("do-indomet-es", "DO_INDOMET", "DOM")),
    ("met.gov.sb", ("sb-sims-en", "SB_SIMS", "SLB")),
    ("inamhi.gob.ec", ("ec-inamhi-es", "EC_INAMHI", "ECU")),
    ("metoffice.gov.tt", ("tt-ttms-en", "TT_TTMS", "TTO")),
    ("meteorology.gov.mv", ("mv-mms-en", "MV_MMS", "MDV")),
    ("meteo.go.tz", ("tz-tma-en", "TZ_TMA", "TZA")),
    ("meteo.bj", ("bj-meteo-fr", "BJ_METEO", "BEN")),
    ("meteobenin", ("bj-meteo-fr", "BJ_METEO", "BEN")),
    ("nms.gov.bz", ("bz-nms-en", "BZ_NMS", "BLZ")),
    ("mtt.gov.bh", ("bh-bmd-en", "BH_BMD", "BHR")),
    ("meteo.co.me", ("me-zhmz-sr", "ME_ZHMZ", "MNE")),
    ("meteolux.lu", ("lu-meteolux-fr", "LU_METEOLUX", "LUX")),
    ("meteorologia.gov.py", ("py-dmh-es", "PY_DMH", "PRY")),
    ("meteotchad", ("td-anam-fr", "TD_ANAM", "TCD")),
    ("mot.gov.jo", ("jo-jmd-en", "JO_JMD", "JOR")),
    ("jmd.gov.jo", ("jo-jmd-en", "JO_JMD", "JOR")),
    ("meteo.gov.vu", ("vu-vmgd-en", "VU_VMGD", "VUT")),
    ("metservice.gov.jm", ("jm-msj-en", "JM_MSJ", "JAM")),
    ("jma.go.jp", ("jp-jma-en", "JP_JMA", "JPN")),
    ("inam.gov.mz", ("mz-inam-pt", "MZ_INAM", "MOZ")),
    ("previsaoinam", ("mz-inam-pt", "MZ_INAM", "MOZ")),
    ("irimo.ir", ("ir-irimo-fa", "IR_IRIMO", "IRN")),
    ("meteo.gov.lk", ("lk-dom-en", "LK_DOM", "LKA")),
    ("meteo_togo", ("tg-anmet-fr", "TG_ANMET", "TGO")),
    ("meteotogo", ("tg-anmet-fr", "TG_ANMET", "TGO")),
    ("nimet.gov.ng", ("ng-nimet-en", "NG_NIMET", "NGA")),
    ("imhpa.gob.pa", ("pa-imhpa-es", "PA_IMHPA", "PAN")),
    ("hidromet.com.pa", ("pa-imhpa-es", "PA_IMHPA", "PAN")),
    ("imn.ac.cr", ("cr-imn-es", "CR_IMN", "CRI")),
    ("dom.org.cy", ("cy-dom-el", "CY_DOM", "CYP")),
    ("namem.gov.mn", ("mn-namem-mn", "MN_NAMEM", "MNG")),
    ("meteo.md", ("md-shs-ro", "MD_SHS", "MDA")),
    ("qweather.gov.qa", ("qa-qmd-en", "QA_QMD", "QAT")),
    ("caa.gov.qa", ("qa-qmd-en", "QA_QMD", "QAT")),
    ("eswatinimet", ("sz-met-en", "SZ_MET", "SWZ")),
    ("meteo.gov.gh", ("gh-gmet-en", "GH_GMET", "GHA")),
    ("inmgb", ("gw-inmgb-pt", "GW_INMGB", "GNB")),
    ("meteomadagascar.mg", ("mg-dgm-fr", "MG_DGM", "MDG")),
    ("meteo.gov.sc", ("sc-sma-en", "SC_SMA", "SYC")),
    ("barbadosweather.org", ("bb-bms-en", "BB_BMS", "BRB")),
    ("meteo.tn", ("tn-inm-fr", "TN_INM", "TUN")),
    ("ersad.gov.sd", ("sd-sma-en", "SD_SMA", "SDN")),
    ("sudanmet", ("sd-sma-en", "SD_SMA", "SDN")),
    ("metmalawi.gov.mw", ("mw-dccms-en", "MW_DCCMS", "MWI")),
    ("meteo.burundi", ("bi-igebu-fr", "BI_IGEBU", "BDI")),
    ("igebu", ("bi-igebu-fr", "BI_IGEBU", "BDI")),
    ("malimeteo.ml", ("ml-malimeteo-fr", "ML_MALIMETEO", "MLI")),
    ("nchmf.gov.vn", ("vn-nchmf-xx", "VN_NCHMF", "VNM")),
    ("anacim.sn", ("sn-anacim-fr", "SN_ANACIM", "SEN")),
    ("hydromet.gov.gy", ("gy-hydromet-en", "GY_HYDROMET", "GUY")),
    ("tmd.go.th", ("th-tmd-en", "TH_TMD", "THA")),
    ("met.gov.kw", ("kw-met-en", "KW_MET", "KWT")),
    ("meteo.rwanda", ("rw-meteo-en", "RW_METEO", "RWA")),
    ("mwe.go.ug", ("ug-mwe-en", "UG_MWE", "UGA")),
    ("unma.go.ug", ("ug-unma-en", "UG_UNMA", "UGA")),
    ("inamet.gov.ao", ("ao-inamet-pt", "AO_INAMET", "AGO")),
    ("senamhi.gob.pe", ("pe-senamhi-es", "PE_SENAMHI", "PER")),
    ("senamhi.gob.bo", ("bo-senamhi-es", "BO_SENAMHI", "BOL")),
    ("dhm.gov.np", ("np-dhm-en", "NP_DHM", "NPL")),
    ("ffwc.gov.bd", ("bd-ffwc-en", "BD_FFWC", "BGD")),
    ("bmd.gov.bd", ("bd-bmd-en", "BD_BMD", "BGD")),
    ("pmd.gov.pk", ("pk-pmd-en", "PK_PMD", "PAK")),
    ("met.gov.my", ("my-met-en", "MY_MET", "MYS")),
    ("insivumeh.gob.gt", ("gt-insivumeh-es", "GT_INSIVUMEH", "GTM")),
    ("google-public-alerts", ("google-public-alerts", "GOOGLE_PUBLIC_ALERTS", "Google")),
]

# Optional local Tier-2 sender mapping loaded from CAP_GAP_SENDER_MAP_JSON (never embedded in public code).
_OPTIONAL_SENDER_MAP_CACHE: dict[str, tuple[str, str, str]] | None = None


def _load_optional_sender_map() -> dict[str, tuple[str, str, str]]:
    global _OPTIONAL_SENDER_MAP_CACHE
    if _OPTIONAL_SENDER_MAP_CACHE is not None:
        return _OPTIONAL_SENDER_MAP_CACHE
    map_path = os.environ.get("CAP_GAP_SENDER_MAP_JSON", "").strip()
    loaded: dict[str, tuple[str, str, str]] = {}
    if map_path and Path(map_path).is_file():
        raw = json.loads(Path(map_path).read_text(encoding="utf-8"))
        for k, v in raw.items():
            if isinstance(v, (list, tuple)) and len(v) == 3:
                loaded[str(k).strip().lower()] = (str(v[0]), str(v[1]), str(v[2]))
    _OPTIONAL_SENDER_MAP_CACHE = loaded
    return loaded


FAH_OID_MAP = {
    "4.1": ("af-andma-en", "AF_ANDMA", "AF"),
    "8.0": ("al-igewe-en", "AL_IGEWE", "AL"),
    "12.0": ("dz-meteo-en", "DZ_METEO", "DZ"),
    "20.0": ("ad-meteo-xx", "AD_METEO", "AD"),
    "24.0": ("ao-inamet-en", "AO_INAMET", "AO"),
    "660.0": ("ai-dma-en", "AI_DMA", "AI"),
    "28.0": ("ag-ms-en", "AG_MS", "AG"),
    "32.0": ("ar-smn-es", "AR_SMN", "AR"),
    "528.3": ("aw-gov-en", "AW_GOV", "AW"),
    "36.0": ("au-bom-en", "AU_BOM", "AU"),
    "40.0": ("at-zamg-xx", "AT_ZAMG", "AT"),
    "40.1": ("at-ehyd-xx", "AT_EHYD", "AT"),
    "48.0": ("bh-meteo-en", "BH_METEO", "BH"),
    "52.1": ("bb-dem-xx", "BB_DEM", "BB"),
    "112.0": ("by-belhydromet-en", "BY_BELHYDROMET", "BY"),
    "56.0": ("be-irm-en", "BE_IRM", "BE"),
    "84.0": ("bz-nms-en", "BZ_NMS", "BZ"),
    "204.0": ("bj-meteo-fr", "BJ_METEO", "BJ"),
    "70.0": ("ba-fhmzbih-bs", "BA_FHMZBIH", "BA"),
    "70.1": ("ba-rhmzrs-bs", "BA_RHMZRS", "BA"),
    "72.0": ("bw-met-en", "BW_DMS", "BW"),
    "76.0": ("br-inmet-pt", "BR_INMET", "BR"),
    "76.3": ("br-sedec-pt", "BR_SEDEC", "BR"),
    "76.1": ("br-obsis-pt", "BR_OBSIS", "BR"),
    "92.0": ("uk-bviddm-en", "GB_BVIDDM", "GB"),
    "100.0": ("bg-meteo-bg", "BG_METEO", "BG"),
    "854.0": ("bf-meteo-fr", "BF_METEO", "BF"),
    "108.0": ("bi-igebu-fr", "BI_METEO", "BI"),
    "120.0": ("cm-meteo-en", "CM_METEO", "CM"),
    "124.2": ("ca-aema-xx", "CA_AEMA", "CA"),
    "124.0": ("ca-msc-xx", "CA_MSC", "CA"),
    "140.0": ("cf-dmn-fr", "CF_DMN", "CF"),
    "148.0": ("td-anam-fr", "TD_ANAM", "TD"),
    "152.0": ("cl-meteo-es", "CL_METEO", "CL"),
    "156.0": ("cn-cma-xx", "CN_CMA", "CN"),
    "170.1": ("co-ideam-xx", "CO_IDEAM", "CO"),
    "170.2": ("co-ungrd-es", "CO_UNGRD", "CO"),
    "174.0": ("km-anacm-fr", "KM_ANACM", "KM"),
    "178.0": ("cg-anac-fr", "CG_ANAC", "CG"),
    "188.0": ("cr-imn-en", "CR_IMN", "CR"),
    "188.2": ("cr-sinamot-en", "CR_SINAMOT", "CR"),
    "384.0": ("ci-sodexam-fr", "CI_SODEXAM", "CI"),
    "191.0": ("hr-meteo-hr", "HR_METEO", "HR"),
    "531.0": ("cw-meteo-en", "CW_METEO", "CW"),
    "196.0": ("cy-dom-en", "CY_DOM", "CY"),
    "203.0": ("cz-chmi-cs", "CZ_CHMI", "CZ"),
    "180.0": ("cd-mettelsat-fr", "CD_METTELSAT", "CD"),
    "208.0": ("dk-dmi-en", "DK_DMI", "DK"),
    "262.0": ("dj-meteo-fr", "DJ_METEO", "DJ"),
    "214.0": ("do-indomet-es", "DO_INDOMET", "DO"),
    "218.0": ("ec-inamhi-es", "EC_INAMHI", "EC"),
    "818.0": ("eg-ema-en", "EG_EMA", "EG"),
    "826.4": ("uk-env-en", "GB_ENV", "GB"),
    "233.0": ("ee-emhi-et", "EE_EMHI", "EE"),
    "748.0": ("sz-met-en", "SZ_MET", "SZ"),
    "231.0": ("et-emi-en", "ET_EMI", "ET"),
    "498.0": ("md-shs-en", "MD_SHS", "MD"),
    "246.0": ("fi-fmi-en", "FI_FMI", "FI"),
    "250.0": ("fr-meteofrance-xx", "FR_METEOFRANCE", "FR"),
    "266.0": ("ga-dgm-fr", "GA_DGM", "GA"),
    "270.0": ("gm-dwr-en", "GM_DWR", "GM"),
    "276.0": ("de-dwd-en", "DE_DWD", "DE"),
    "288.0": ("gh-gmet-en", "GH_GMET", "GH"),
    "300.0": ("gr-hnms-el", "GR_HNMS", "GR"),
    "324.0": ("gn-anm-fr", "GN_ANM", "GN"),
    "624.0": ("gw-inm-pt", "GW_INM", "GW"),
    "328.0": ("gy-hms-en", "GY_HMS", "GY"),
    "344.0": ("hk-hko-xx", "HK_HKO", "HK"),
    "348.0": ("hu-met-xx", "HU_MET", "HU"),
    "352.0": ("is-vedur-is", "IS_VEDUR", "IS"),
    "356.0": ("in-imd-en", "IN_IMD", "IN"),
    "356.1": ("in-ndma-xx", "IN_NDMA", "IN"),
    "360.0": ("id-bmkg-en", "ID_BMKG", "ID"),
    "364.0": ("ir-irimo-en", "IR_IRIMO", "IR"),
    "368.0": ("iq-imos-en", "IQ_IMOS", "IQ"),
    "372.0": ("ie-met-en", "IE_MET", "IE"),
    "376.0": ("il-ims-xx", "IL_IMS", "IL"),
    "380.3": ("it-meteoam-it", "IT_METEOAM", "IT"),
    "548.0": ("vu-vmgd-en", "VU_VMGD", "VU"),
    "400.0": ("jo-meteo-en", "JO_METEO", "JO"),
    "398.0": ("kz-kazhydromet-en", "KZ_KAZHYDROMET", "KZ"),
    "404.0": ("ke-kmd-en", "KE_KMD", "KE"),
    "414.0": ("kw-dom-en", "KW_DOM", "KW"),
    "417.0": ("kg-meteo-en", "KG_METEO", "KG"),
    "428.0": ("lv-lvgmc-lv", "LV_LVGMC", "LV"),
    "426.0": ("ls-lms-en", "LS_LMS", "LS"),
    "430.0": ("lr-meteo-en", "LR_METEO", "LR"),
    "434.0": ("ly-nmc-en", "LY_NMC", "LY"),
    "440.0": ("lt-lhms-xx", "LT_LHMS", "LT"),
    "442.0": ("lu-ana-meteo-fr", "LU_ANA-METEO", "LU"),
    "446.0": ("mo-smg-xx", "MO_SMG", "MO"),
    "450.0": ("mg-meteo-en", "MG_METEO", "MG"),
    "454.0": ("mw-met-en", "MW_MET", "MW"),
    "462.0": ("mv-met-en", "MV_MET", "MV"),
    "466.0": ("ml-meteo-fr", "ML_METEO", "ML"),
    "470.0": ("mt-maltairport-en", "MT_MALTAIRPORT", "MT"),
    "584.0": ("mh-gov-en", "MH_GOV", "MH"),
    "585.0": ("pw-wso-en", "PW_WSO", "PW"),
    "478.0": ("mr-onm-fr", "MR_ONM", "MR"),
    "480.0": ("mu-mms-en", "MU_MMS", "MU"),
    "484.0": ("mx-smn-es", "MX_SMN", "MX"),
    "496.0": ("mn-namem-en", "MN_NAMEM", "MN"),
    "499.0": ("me-meteo-xx", "ME_METEO", "ME"),
    "508.0": ("mz-inam-pt", "MZ_INAM", "MZ"),
    "528.0": ("nl-rnmi-xx", "NL_RNMI", "NL"),
    "554.1": ("nz-gns-en", "NZ_GEO", "NZ"),
    "554.0": ("nz-nms-en", "NZ_NMS", "NZ"),
    "554.2": ("nz-nema-en", "NZ_NEMA", "NZ"),
    "562.0": ("ne-meteo-fr", "NE_METEO", "NE"),
    "566.0": ("ng-nimet-en", "NG_NIMET", "NG"),
    "570.0": ("nu-gov-en", "NU_GOV", "NU"),
    "807.0": ("mk-meteo-en", "MK_METEO", "MK"),
    "578.0": ("no-met-en", "NO_MET", "NO"),
    "578.1": ("no-nve-xx", "NO_NVE", "NO"),
    "512.0": ("om-met-en", "OM_MET", "OM"),
    "586.0": ("pk-pmd-en", "PK_PMD", "PK"),
    "591.0": ("pa-imhpa-es", "PA_IMHPA", "PA"),
    "600.0": ("py-dmh-es", "PY_DMH", "PY"),
    "608.0": ("ph-pagasa-en", "PH_PAGASA", "PH"),
    "616.0": ("pl-imgw-xx", "PL_IMGW", "PL"),
    "620.0": ("pt-ipma-pt", "PT_IPMA", "PT"),
    "634.0": ("qa-caa-en", "QA_CAA", "QA"),
    "410.0": ("kr-kma-weather-en", "KR_KMA-WEATHER", "KR"),
    "642.0": ("ro-meteoromania-xx", "RO_METEOROMANIA", "RO"),
    "643.0": ("ru-roshydromet-en", "RU_ROSHYDROMET", "RU"),
    "662.0": ("lc-ms-en", "LC_MS", "LC"),
    "670.0": ("vc-ms-en", "VC_MS", "VC"),
    "882.0": ("ws-smd-en", "WS_SMD", "WS"),
    "678.0": ("st-meteo-en", "ST_METEO", "ST"),
    "682.0": ("sa-ncm-en", "SA_NCM", "SA"),
    "686.0": ("sn-anacim-fr", "SN_ANACIM", "SN"),
    "688.0": ("rs-hidmet-sr", "RS_HIDMET", "RS"),
    "690.0": ("sc-meteo-en", "SC_METEO", "SC"),
    "694.0": ("sl-slmet-en", "SL_SLMET", "SL"),
    "702.0": ("sg-mss-en", "SG_MSS", "SG"),
    "702.2": ("sg-pub-en", "SG_PUB", "SG"),
    "703.0": ("sk-shmu-xx", "SK_SHMU", "SK"),
    "705.1": ("si-meteo-xx", "SI_METEO", "SI"),
    "90.0": ("sb-met-en", "SB_MET", "SB"),
    "706.123": ("so-snma-en", "SO_SNMA", "SO"),
    "710.0": ("za-saws-en", "ZA_SAWS", "ZA"),
    "728.0": ("ss-ssms-en", "SS_SSMS", "SS"),
    "724.0": ("es-aemet-es", "ES_AEMET", "ES"),
    "144.0": ("lk-meteo-en", "LK_METEO", "LK"),
    "729.1": ("sd-hydro-en", "SD_HYDRO", "SD"),
    "729.0": ("sd-sma-en", "SD_SMA", "SD"),
    "740.0": ("sr-meteo-en", "SR_METEO", "SR"),
    "752.0": ("se-smhi-xx", "SE_SMHI", "SE"),
    "756.0": ("ch-meteoswiss-de", "CH_METEOSWISS", "CH"),
    "760.1": ("sy-moeadm-ar", "SY_MOEADM", "SY"),
    "834.0": ("tz-tma-en", "TZ_TMA", "TZ"),
    "764.0": ("th-tmd-en", "TH_TMD", "TH"),
    "626.0": ("tl-dnmg-en", "TL_DNMG", "TL"),
    "768.0": ("tg-anamet-fr", "TG_DGMN", "TG"),
    "776.0": ("to-tms-xx", "TO_TMS", "TO"),
    "780.0": ("tt-ttms-en", "TT_TTMS", "TT"),
    "788.0": ("tn-meteo-en", "TN_METEO", "TN"),
    "796.2": ("tc-gov-en", "TC_GOV", "TC"),
    "798.0": ("tv-tuvmet-en", "TV_TUVMET", "TV"),
    "800.0": ("ug-dms-en", "UG_DMS", "UG"),
    "804.0": ("ua-meteo-en", "UA_METEO", "UA"),
    "826.0": ("uk-metoffice-en", "GB_METOFFICE", "GB"),
    "840.3": ("us-fema-ipaws-en", "US_FEMA-IPAWS", "US"),
    "840.1": ("us-noaa-ptwc-en", "US_NOAA-PTWC", "US"),
    "840.0": ("us-noaa-mzus-en", "US_NOAA-MZUS", "US"),
    "840.5": ("us-usgs-vhp-en", "US_USGS-VHP", "US"),
    "858.0": ("uy-inumet-es", "UY_INUMET", "UY"),
    "860.0": ("uz-hydromet-en", "UZ_HYDROMET", "UZ"),
    "704.0": ("vn-nchmf-xx", "VN_NCHMF", "VN"),
    "887.0": ("ye-yms-en", "YE_YMS", "YE"),
    "894.0": ("zm-zmd-en", "ZM_ZMD", "ZM"),
    "716.0": ("zw-msd-en", "ZW_MSD", "ZW"),
}
FAH_ISO2_DEFAULT = {
    "AF": ("af-andma-en", "AF_ANDMA"),
    "AL": ("al-igewe-en", "AL_IGEWE"),
    "DZ": ("dz-meteo-en", "DZ_METEO"),
    "AD": ("ad-meteo-xx", "AD_METEO"),
    "AO": ("ao-inamet-en", "AO_INAMET"),
    "AI": ("ai-dma-en", "AI_DMA"),
    "AG": ("ag-ms-en", "AG_MS"),
    "AR": ("ar-smn-es", "AR_SMN"),
    "AW": ("aw-gov-en", "AW_GOV"),
    "AU": ("au-sa-cfs-en", "AU_SA-CFS"),
    "AT": ("at-zamg-xx", "AT_ZAMG"),
    "BH": ("bh-meteo-en", "BH_METEO"),
    "BB": ("bb-dem-xx", "BB_DEM"),
    "BY": ("by-belhydromet-en", "BY_BELHYDROMET"),
    "BE": ("be-irm-en", "BE_IRM"),
    "BZ": ("bz-nms-en", "BZ_NMS"),
    "BJ": ("bj-meteo-fr", "BJ_METEO"),
    "BA": ("ba-fhmzbih-bs", "BA_FHMZBIH"),
    "BW": ("bw-met-en", "BW_DMS"),
    "BR": ("br-inmet-pt", "BR_INMET"),
    "GB": ("uk-metoffice-en", "GB_METOFFICE"),
    "BG": ("bg-meteo-bg", "BG_METEO"),
    "BF": ("bf-meteo-fr", "BF_METEO"),
    "BI": ("bi-igebu-fr", "BI_METEO"),
    "CM": ("cm-meteo-en", "CM_METEO"),
    "CA": ("ca-aema-xx", "CA_AEMA"),
    "CF": ("cf-dmn-fr", "CF_DMN"),
    "TD": ("td-anam-fr", "TD_ANAM"),
    "CL": ("cl-meteo-es", "CL_METEO"),
    "CN": ("cn-cma-xx", "CN_CMA"),
    "CO": ("co-ideam-xx", "CO_IDEAM"),
    "KM": ("km-anacm-fr", "KM_ANACM"),
    "CG": ("cg-anac-fr", "CG_ANAC"),
    "CR": ("cr-sinamot-en", "CR_SINAMOT"),
    "CI": ("ci-sodexam-fr", "CI_SODEXAM"),
    "HR": ("hr-meteo-hr", "HR_METEO"),
    "CW": ("cw-meteo-en", "CW_METEO"),
    "CY": ("cy-dom-en", "CY_DOM"),
    "CZ": ("cz-chmi-cs", "CZ_CHMI"),
    "CD": ("cd-mettelsat-fr", "CD_METTELSAT"),
    "DK": ("dk-dmi-en", "DK_DMI"),
    "DJ": ("dj-meteo-fr", "DJ_METEO"),
    "DO": ("do-indomet-es", "DO_INDOMET"),
    "EC": ("ec-inamhi-es", "EC_INAMHI"),
    "EG": ("eg-ema-en", "EG_EMA"),
    "EE": ("ee-emhi-et", "EE_EMHI"),
    "SZ": ("sz-met-en", "SZ_MET"),
    "ET": ("et-emi-en", "ET_EMI"),
    "FM": ("fm-wsoy-en", "FM_WSOY"),
    "FI": ("fi-fmi-en", "FI_FMI"),
    "FR": ("fr-meteofrance-xx", "FR_METEOFRANCE"),
    "GA": ("ga-dgm-fr", "GA_DGM"),
    "GM": ("gm-dwr-en", "GM_DWR"),
    "DE": ("de-dwd-en", "DE_DWD"),
    "GH": ("gh-gmet-en", "GH_GMET"),
    "GR": ("gr-hnms-el", "GR_HNMS"),
    "GN": ("gn-anm-fr", "GN_ANM"),
    "GW": ("gw-inm-pt", "GW_INM"),
    "GY": ("gy-hms-en", "GY_HMS"),
    "HK": ("hk-hko-xx", "HK_HKO"),
    "HU": ("hu-met-xx", "HU_MET"),
    "IS": ("is-vedur-is", "IS_VEDUR"),
    "IN": ("in-imd-en", "IN_IMD"),
    "ID": ("id-bmkg-en", "ID_BMKG"),
    "INT": ("int-gdacs-en", "INT_GDACS"),
    "IR": ("ir-irimo-en", "IR_IRIMO"),
    "IQ": ("iq-imos-en", "IQ_IMOS"),
    "IE": ("ie-met-en", "IE_MET"),
    "IL": ("il-ims-xx", "IL_IMS"),
    "IT": ("it-meteoam-it", "IT_METEOAM"),
    "JM": ("jm-jms-en", "JM_JMS"),
    "JO": ("jo-meteo-en", "JO_METEO"),
    "KZ": ("kz-kazhydromet-en", "KZ_KAZHYDROMET"),
    "KE": ("ke-kmd-en", "KE_KMD"),
    "KW": ("kw-dom-en", "KW_DOM"),
    "KG": ("kg-meteo-en", "KG_METEO"),
    "LV": ("lv-lvgmc-lv", "LV_LVGMC"),
    "LS": ("ls-lms-en", "LS_LMS"),
    "LR": ("lr-meteo-en", "LR_METEO"),
    "LY": ("ly-nmc-en", "LY_NMC"),
    "LT": ("lt-lhms-xx", "LT_LHMS"),
    "LU": ("lu-ana-meteo-fr", "LU_ANA-METEO"),
    "MO": ("mo-smg-xx", "MO_SMG"),
    "MG": ("mg-meteo-en", "MG_METEO"),
    "MW": ("mw-met-en", "MW_MET"),
    "MV": ("mv-met-en", "MV_MET"),
    "ML": ("ml-meteo-fr", "ML_METEO"),
    "MT": ("mt-maltairport-en", "MT_MALTAIRPORT"),
    "MH": ("mh-wso-en", "MH_WSO"),
    "MR": ("mr-onm-fr", "MR_ONM"),
    "MU": ("mu-mms-en", "MU_MMS"),
    "MX": ("mx-smn-es", "MX_SMN"),
    "MD": ("md-shs-en", "MD_SHS"),
    "MN": ("mn-namem-en", "MN_NAMEM"),
    "ME": ("me-meteo-xx", "ME_METEO"),
    "MZ": ("mz-inam-pt", "MZ_INAM"),
    "NL": ("nl-rnmi-xx", "NL_RNMI"),
    "NZ": ("nz-nema-en", "NZ_NEMA"),
    "NE": ("ne-meteo-fr", "NE_METEO"),
    "NG": ("ng-nimet-en", "NG_NIMET"),
    "NU": ("nu-gov-en", "NU_GOV"),
    "MK": ("mk-meteo-en", "MK_METEO"),
    "NO": ("no-met-en", "NO_MET"),
    "OM": ("om-met-en", "OM_MET"),
    "PK": ("pk-pmd-en", "PK_PMD"),
    "PW": ("pw-wso-en", "PW_WSO"),
    "PA": ("pa-imhpa-es", "PA_IMHPA"),
    "PY": ("py-dmh-es", "PY_DMH"),
    "PH": ("ph-pagasa-en", "PH_PAGASA"),
    "PL": ("pl-imgw-xx", "PL_IMGW"),
    "PT": ("pt-ipma-pt", "PT_IPMA"),
    "QA": ("qa-caa-en", "QA_CAA"),
    "KR": ("kr-kma-weather-en", "KR_KMA-WEATHER"),
    "RO": ("ro-meteoromania-xx", "RO_METEOROMANIA"),
    "RU": ("ru-roshydromet-en", "RU_ROSHYDROMET"),
    "LC": ("lc-ms-en", "LC_MS"),
    "VC": ("vc-ms-en", "VC_MS"),
    "WS": ("ws-smd-en", "WS_SMD"),
    "ST": ("st-meteo-en", "ST_METEO"),
    "SA": ("sa-ncm-en", "SA_NCM"),
    "SN": ("sn-anacim-fr", "SN_ANACIM"),
    "RS": ("rs-hidmet-sr", "RS_HIDMET"),
    "SC": ("sc-meteo-en", "SC_METEO"),
    "SL": ("sl-slmet-en", "SL_SLMET"),
    "SG": ("sg-pub-en", "SG_PUB"),
    "SK": ("sk-shmu-xx", "SK_SHMU"),
    "SI": ("si-meteo-xx", "SI_METEO"),
    "SB": ("sb-met-en", "SB_MET"),
    "SO": ("so-snma-en", "SO_SNMA"),
    "ZA": ("za-saws-en", "ZA_SAWS"),
    "SS": ("ss-ssms-en", "SS_SSMS"),
    "ES": ("es-aemet-es", "ES_AEMET"),
    "LK": ("lk-meteo-en", "LK_METEO"),
    "SD": ("sd-sma-en", "SD_SMA"),
    "SR": ("sr-meteo-en", "SR_METEO"),
    "SE": ("se-smhi-xx", "SE_SMHI"),
    "CH": ("ch-meteoswiss-de", "CH_METEOSWISS"),
    "SY": ("sy-moeadm-ar", "SY_MOEADM"),
    "TW": ("tw-ncdr-zh", "TW_NCDR"),
    "TZ": ("tz-tma-en", "TZ_TMA"),
    "TH": ("th-tmd-en", "TH_TMD"),
    "TL": ("tl-dnmg-en", "TL_DNMG"),
    "TG": ("tg-anamet-fr", "TG_DGMN"),
    "TO": ("to-tms-xx", "TO_TMS"),
    "TT": ("tt-ttms-en", "TT_TTMS"),
    "TN": ("tn-meteo-en", "TN_METEO"),
    "TC": ("tc-gov-en", "TC_GOV"),
    "TV": ("tv-tuvmet-en", "TV_TUVMET"),
    "UG": ("ug-dms-en", "UG_DMS"),
    "UA": ("ua-meteo-en", "UA_METEO"),
    "US": ("us-usgs-vhp-en", "US_USGS-VHP"),
    "UY": ("uy-inumet-es", "UY_INUMET"),
    "UZ": ("uz-hydromet-en", "UZ_HYDROMET"),
    "VU": ("vu-vmgd-en", "VU_VMGD"),
    "VN": ("vn-nchmf-xx", "VN_NCHMF"),
    "YE": ("ye-yms-en", "YE_YMS"),
    "ZM": ("zm-zmd-en", "ZM_ZMD"),
    "ZW": ("zw-msd-en", "ZW_MSD"),
}
FAH_DOMAIN_MAP = {
    "ametvigilance.meteo.dz": ("dz-meteo-en", "DZ_METEO", "DZ"),
    "govern.ad": ("ad-meteo-xx", "AD_METEO", "AD"),
    "inamet.gov.ao": ("ao-inamet-pt", "AO_INAMET", "AO"),
    "gov.ai": ("ai-dma-en", "AI_DMA", "AI"),
    "axa-primary.capews.com": ("ai-dma-en", "AI_DMA", "AI"),
    "atg-primary.capews.com": ("ag-ms-en", "AG_MS", "AG"),
    "smn.gov.ar": ("ar-smn-es", "AR_SMN", "AR"),
    "ssl.smn.gob.ar": ("ar-smn-es", "AR_SMN", "AR"),
    "crisis.aw": ("aw-gov-es", "AW_GOV", "AW"),
    "bom.gov.au": ("au-bom-en", "AU_BOM", "AU"),
    "severeweather.wmo.int": ("au-bom-en", "AU_BOM", "AU"),
    "sa.gov.au": ("au-sa-cfs-en", "AU_SA-CFS", "AU"),
    "data.eso.sa.gov.au": ("au-sa-cfs-en", "AU_SA-CFS", "AU"),
    "zamg.ac.at": ("at-ehyd-xx", "AT_EHYD", "AT"),
    "mtt.gov.bh": ("bh-meteo-en", "BH_METEO", "BH"),
    "barbados.gov.bb": ("bb-dem-xx", "BB_DEM", "BB"),
    "brb-secondary.capews.com": ("bb-dem-xx", "BB_DEM", "BB"),
    "hmc.by": ("by-belhydromet-ru", "BY_BELHYDROMET", "BY"),
    "meteoalert.meteoinfo.ru": ("uz-hydromet-en", "UZ_HYDROMET", "UZ"),
    "geosphere.at": ("ch-meteoswiss-de", "CH_METEOSWISS", "CH"),
    "nms.gov.bz": ("bz-nms-es", "BZ_NMS", "BZ"),
    "meteobenin.bj": ("bj-meteo-fr", "BJ_METEO", "BJ"),
    "fhmzbih.gov.ba": ("ba-fhmzbih-bs", "BA_FHMZBIH", "BA"),
    "gov.bw": ("bw-met-en", "BW_DMS", "BW"),
    "inmet.gov.br": ("br-sedec-pt", "BR_SEDEC", "BR"),
    "apiprevmet3.inmet.gov.br": ("br-inmet-pt", "BR_INMET", "BR"),
    "idapfile.mdr.gov.br": ("br-sedec-pt", "BR_SEDEC", "BR"),
    "unb.br": ("br-obsis-pt", "BR_OBSIS", "BR"),
    "obsis.unb.br": ("br-obsis-pt", "BR_OBSIS", "BR"),
    "gov.vg": ("uk-bviddm-en", "GB_BVIDDM", "GB"),
    "meteoburkina.bf": ("bf-meteo-fr", "BF_METEO", "BF"),
    "igebu.bi": ("bi-igebu-fr", "BI_METEO", "BI"),
    "gov.ab.ca": ("ca-aema-xx", "CA_AEMA", "CA"),
    "alberta.ca": ("ca-aema-xx", "CA_AEMA", "CA"),
    "ec.gc.ca": ("ca-msc-xx", "CA_MSC", "CA"),
    "rss.naad-adna.pelmorex.com": ("ca-msc-xx", "CA_MSC", "CA"),
    "meteotchad.org": ("td-anam-fr", "TD_ANAM", "TD"),
    "dgac.gob.cl": ("cl-meteo-es", "CL_METEO", "CL"),
    "archivos.meteochile.gob.cl": ("cl-meteo-es", "CL_METEO", "CL"),
    "12379.cn": ("cn-cma-xx", "CN_CMA", "CN"),
    "alert-feed-worldweather-org.s3.amazonaws.com": ("cn-cma-xx", "CN_CMA", "CN"),
    "ideam.gov.co": ("co-ideam-xx", "CO_IDEAM", "CO"),
    "alertascap.ideam.gov.co": ("co-ideam-xx", "CO_IDEAM", "CO"),
    "gestiondelriesgo.gov.co": ("co-ungrd-es", "CO_UNGRD", "CO"),
    "meteocomores.km": ("km-anacm-fr", "KM_ANACM", "KM"),
    "dirmet.cg": ("cg-anac-fr", "CG_ANAC", "CG"),
    "imn.ac.cr": ("cr-imn-es", "CR_IMN", "CR"),
    "una.ac.cr": ("cr-sinamot-es", "CR_SINAMOT", "CR"),
    "cirus.dhz.hr": ("hr-meteo-hr", "HR_METEO", "HR"),
    "meteo.cw": ("cw-meteo-pap", "CW_METEO", "CW"),
    "chmi.cz": ("cz-chmi-cs", "CZ_CHMI", "CZ"),
    "meteordcongo.cd": ("cd-mettelsat-fr", "CD_METTELSAT", "CD"),
    "dmi.dk": ("dk-dmi-en", "DK_DMI", "DK"),
    "meteodjibouti.dj": ("dj-meteo-fr", "DJ_METEO", "DJ"),
    "indomet.gov.do": ("do-indomet-es", "DO_INDOMET", "DO"),
    "inamhi.gob.ec": ("ec-inamhi-es", "EC_INAMHI", "EC"),
    "environment-agency.gov.uk": ("uk-env-en", "GB_ENV", "GB"),
    "environment.data.gov.uk": ("uk-env-en", "GB_ENV", "GB"),
    "ethiomet.gov.et": ("et-emi-en", "ET_EMI", "ET"),
    "noaa.gov": ("us-noaa-ptwc-en", "US_NOAA-PTWC", "US"),
    "api.weather.gov": ("us-noaa-mzus-en", "US_NOAA-MZUS", "US"),
    "fmi.fi": ("fi-fmi-en", "FI_FMI", "FI"),
    "alerts.fmi.fi": ("fi-fmi-en", "FI_FMI", "FI"),
    "meteo.fr": ("fr-meteofrance-xx", "FR_METEOFRANCE", "FR"),
    "meteogambia.gm": ("gm-dwr-en", "GM_DWR", "GM"),
    "dwd.de": ("de-dwd-en", "DE_DWD", "DE"),
    "meteo.gov.gh": ("gh-gmet-en", "GH_GMET", "GH"),
    "hnms.gr": ("gr-hnms-el", "GR_HNMS", "GR"),
    "anmeteo.gov.gn": ("gn-anm-fr", "GN_ANM", "GN"),
    "meteoguinebissau.gw": ("gw-inm-pt", "GW_INM", "GW"),
    "hko.gov.hk": ("hk-hko-xx", "HK_HKO", "HK"),
    "alerts.weather.gov.hk": ("hk-hko-xx", "HK_HKO", "HK"),
    "met.hu": ("hu-met-xx", "HU_MET", "HU"),
    "vedur.is": ("is-vedur-is", "IS_VEDUR", "IS"),
    "api.vedur.is": ("is-vedur-is", "IS_VEDUR", "IS"),
    "imd.gov.in": ("in-imd-en", "IN_IMD", "IN"),
    "ndma.gov.in": ("in-ndma-xx", "IN_NDMA", "IN"),
    "sachet.ndma.gov.in": ("in-ndma-xx", "IN_NDMA", "IN"),
    "bmkg.go.id": ("id-bmkg-en", "ID_BMKG", "ID"),
    "bmkg-content-inatews.storage.googleapis.com": ("id-inatews-xx", "ID_INATEWS", "ID"),
    "ec.europa.eu": ("int-gdacs-en", "INT_GDACS", "INT"),
    "gdacs.org": ("int-gdacs-en", "INT_GDACS", "INT"),
    "met.ie": ("ie-met-en", "IE_MET", "IE"),
    "ims.gov.il": ("il-ims-xx", "IL_IMS", "IL"),
    "aeronautica.difesa.it": ("it-meteoam-it", "IT_METEOAM", "IT"),
    "metservice.gov.jm": ("jm-jms-en", "JM_JMS", "JM"),
    "alert.metservice.gov.jm": ("jm-jms-en", "JM_JMS", "JM"),
    "mot.gov.jo": ("jo-meteo-en", "JO_METEO", "JO"),
    "meteo.kz": ("kz-kazhydromet-ru", "KZ_KAZHYDROMET", "KZ"),
    "meteo.go.ke": ("ke-kmd-en", "KE_KMD", "KE"),
    "dgca.gov.kw": ("kw-dom-en", "KW_DOM", "KW"),
    "met.gov.kw": ("kw-dom-en", "KW_DOM", "KW"),
    "meteo.kg": ("kg-meteo-en", "KG_METEO", "KG"),
    "lvgmc.lv": ("lv-lvgmc-lv", "LV_LVGMC", "LV"),
    "gov.ls": ("ls-lms-en", "LS_LMS", "LS"),
    "lesmet.gov.ls": ("ls-lms-en", "LS_LMS", "LS"),
    "meteoliberia.com": ("lr-meteo-en", "LR_METEO", "LR"),
    "meteo.lt": ("lt-lhms-xx", "LT_LHMS", "LT"),
    "smg.gov.mo": ("mo-smg-xx", "MO_SMG", "MO"),
    "rss.smg.gov.mo": ("mo-smg-xx", "MO_SMG", "MO"),
    "metmalawi.gov.mw": ("mw-met-en", "MW_MET", "MW"),
    "met.gov.mv": ("mv-met-en", "MV_MET", "MV"),
    "cap.meteorology.gov.mv": ("mv-met-en", "MV_MET", "MV"),
    "malimeteo.ml": ("ml-meteo-fr", "ML_METEO", "ML"),
    "maltairport.com": ("mt-maltairport-en", "MT_MALTAIRPORT", "MT"),
    "rmigov.mh": ("mh-wso-en", "MH_WSO", "MH"),
    "meteomauritanie.mr": ("mr-onm-fr", "MR_ONM", "MR"),
    "conagua.gob.mx": ("mx-smn-es", "MX_SMN", "MX"),
    "smn.conagua.gob.mx": ("mx-smn-es", "MX_SMN", "MX"),
    "namem.gov.mn": ("mn-namem-mn", "MN_NAMEM", "MN"),
    "meteo.co.me": ("me-meteo-xx", "ME_METEO", "ME"),
    "inam.gov.mz": ("mz-inam-pt", "MZ_INAM", "MZ"),
    "knmi.nl": ("nl-rnmi-xx", "NL_RNMI", "NL"),
    "gns.cri.nz": ("nz-gns-en", "NZ_GEO", "NZ"),
    "api.geonet.org.nz": ("nz-gns-en", "NZ_GEO", "NZ"),
    "metservice.com": ("nz-nms-en", "NZ_NMS", "NZ"),
    "alerts.metservice.com": ("nz-nms-en", "NZ_NMS", "NZ"),
    "nema.govt.nz": ("nz-nema-en", "NZ_NEMA", "NZ"),
    "alerthub.civildefence.govt.nz": ("nz-nema-en", "NZ_NEMA", "NZ"),
    "niger-meteo.ne": ("ne-meteo-fr", "NE_METEO", "NE"),
    "nimet.gov.ng": ("ng-nimet-ha", "NG_NIMET", "NG"),
    "gov.nu": ("nu-gov-en", "NU_GOV", "NU"),
    "met.no": ("no-met-en", "NO_MET", "NO"),
    "alert.met.no": ("no-met-en", "NO_MET", "NO"),
    "nve.no": ("no-nve-xx", "NO_NVE", "NO"),
    "cap.nve.no": ("no-nve-xx", "NO_NVE", "NO"),
    "caa.gov.om": ("om-met-en", "OM_MET", "OM"),
    "imhpa.gob.pa": ("pa-imhpa-es", "PA_IMHPA", "PA"),
    "meteorologia.gov.py": ("py-dmh-es", "PY_DMH", "PY"),
    "publicalert.pagasa.dost.gov.ph": ("ph-pagasa-en", "PH_PAGASA", "PH"),
    "imgw.pl": ("pl-imgw-xx", "PL_IMGW", "PL"),
    "ipma.pt": ("pt-ipma-pt", "PT_IPMA", "PT"),
    "caa.gov.qa": ("qa-caa-en", "QA_CAA", "QA"),
    "weather.go.kr": ("kr-kma-weather-en", "KR_KMA-WEATHER", "KR"),
    "meteoromania.ro": ("ro-meteoromania-xx", "RO_METEOROMANIA", "RO"),
    "mecom.ru": ("ru-roshydromet-en", "RU_ROSHYDROMET", "RU"),
    "meteoinfo.ru": ("ru-roshydromet-en", "RU_ROSHYDROMET", "RU"),
    "govt.lc": ("lc-ms-en", "LC_MS", "LC"),
    "saint-lucia.cap-composer.wiscaribbeancmo.org": ("lc-ms-en", "LC_MS", "LC"),
    "vct-secondary.capews.com": ("vc-ms-en", "VC_MS", "VC"),
    "alert.samet.gov.ws": ("ws-smd-en", "WS_SMD", "WS"),
    "ncm.gov.sa": ("sa-ncm-en", "SA_NCM", "SA"),
    "anacim.sn": ("sn-anacim-fr", "SN_ANACIM", "SN"),
    "hidmet.gov.rs": ("rs-hidmet-sr", "RS_HIDMET", "RS"),
    "meteo.gov.sc": ("sc-meteo-en", "SC_METEO", "SC"),
    "meteo.sc": ("sc-meteo-en", "SC_METEO", "SC"),
    "slmet.gov.sl": ("sl-slmet-en", "SL_SLMET", "SL"),
    "nea.gov.sg": ("sg-mss-en", "SG_MSS", "SG"),
    "weather.gov.sg": ("sg-mss-en", "SG_MSS", "SG"),
    "pub.gov.sg": ("sg-pub-en", "SG_PUB", "SG"),
    "d2x63xuv21a9h4.cloudfront.net": ("sg-pub-en", "SG_PUB", "SG"),
    "shmu.sk": ("sk-shmu-xx", "SK_SHMU", "SK"),
    "gov.si": ("si-meteo-xx", "SI_METEO", "SI"),
    "meteosomalia.so": ("so-snma-en", "SO_SNMA", "SO"),
    "weathersa.co.za": ("za-saws-en", "ZA_SAWS", "ZA"),
    "caps.weathersa.co.za": ("za-saws-en", "ZA_SAWS", "ZA"),
    "meteosouthsudan.com.ss": ("ss-ssms-en", "SS_SSMS", "SS"),
    "aemet.es": ("es-aemet-es", "ES_AEMET", "ES"),
    "was.meteo.gov.lk": ("lk-meteo-ta", "LK_METEO", "LK"),
    "hydrosudan.sd": ("sd-hydro-en", "SD_HYDRO", "SD"),
    "ersad.gov.sd": ("sd-sma-en", "SD_SMA", "SD"),
    "meteosudan.sd": ("sd-sma-en", "SD_SMA", "SD"),
    "med.gov.sy": ("sy-moeadm-ar", "SY_MOEADM", "SY"),
    "climweb.med.gov.sy": ("sy-moeadm-ar", "SY_MOEADM", "SY"),
    "ncdr.nat.gov.tw": ("tw-ncdr-zh", "TW_NCDR", "TW"),
    "alerts.ncdr.nat.gov.tw": ("tw-ncdr-zh", "TW_NCDR", "TW"),
    "meteo.go.tz": ("tz-tma-sw", "TZ_TMA", "TZ"),
    "tmd.go.th": ("th-tmd-th", "TH_TMD", "TH"),
    "ms-api.dnmg.gov.tl": ("tl-dnmg-en", "TL_DNMG", "TL"),
    "meteotogo.tg": ("tg-anamet-fr", "TG_DGMN", "TG"),
    "met.gov.to": ("to-tms-xx", "TO_TMS", "TO"),
    "alerts.met.gov.to": ("to-tms-xx", "TO_TMS", "TO"),
    "gov.tt": ("tt-ttms-en", "TT_TTMS", "TT"),
    "metproducts.gov.tt": ("tt-ttms-en", "TT_TTMS", "TT"),
    "meteo.tn": ("tn-meteo-fr", "TN_METEO", "TN"),
    "tciairports.tc": ("tc-gov-es", "TC_GOV", "TC"),
    "gov.tv": ("tv-tuvmet-en", "TV_TUVMET", "TV"),
    "wids.mwe.go.ug": ("ug-dms-en", "UG_DMS", "UG"),
    "metoffice.gov.uk": ("uk-metoffice-en", "GB_METOFFICE", "GB"),
    "airnow.zendesk.com": ("us-epa-aq-en", "US_EPA-AQ", "US"),
    "feeds.enviroflash.info": ("us-epa-aq-en", "US_EPA-AQ", "US"),
    "fema.dhs.gov": ("us-fema-ipaws-en", "US_FEMA-IPAWS", "US"),
    "apps.fema.gov": ("us-fema-ipaws-en", "US_FEMA-IPAWS", "US"),
    "tsunami.gov": ("us-noaa-ptwc-en", "US_NOAA-PTWC", "US"),
    "usgs.gov": ("us-usgs-vhp-en", "US_USGS-VHP", "US"),
    "volcanoes.usgs.gov": ("us-usgs-vhp-en", "US_USGS-VHP", "US"),
    "inumet.gub.uy": ("uy-inumet-es", "UY_INUMET", "UY"),
    "meteo.gov.vu": ("vu-vmgd-en", "VU_VMGD", "VU"),
    "smartalert.nchmf.gov.vn": ("vn-nchmf-xx", "VN_NCHMF", "VN"),
    "zmd.gov.zm": ("zm-zmd-en", "ZM_ZMD", "ZM"),
    "weatherzw.org.zw": ("zw-msd-en", "ZW_MSD", "ZW"),
}


def make_canonical_alert_id(resolved_feed_id, sender, identifier) -> str:
    """Derives a provenance-scoped deterministic UUID5 key shared across all CAP database writers."""
    feed_part = str(resolved_feed_id or "unmapped-feed").strip().lower()
    sender_part = str(sender or "").strip().lower()
    id_part = str(identifier or "").strip()
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{feed_part}:{sender_part}:{id_part}"))


def _extract_sender_host_and_path(s_low: str) -> tuple[str, str]:
    """Extracts normalized domain/hostname and optional path from an email, URL, or bare host."""
    if not s_low:
        return "", ""
    if "://" in s_low:
        parsed = urlparse(s_low)
        return (parsed.hostname or "").rstrip("."), parsed.path or ""
    if "@" in s_low:
        _, domain_part = s_low.rsplit("@", 1)
        domain_clean = domain_part.strip().split()[0].strip("<>.,;:/")
        return domain_clean.rstrip("."), ""
    token = s_low.split()[0].strip("<>.,;")
    if "/" in token:
        host_part, path_part = token.split("/", 1)
        return host_part.rstrip("."), "/" + path_part
    if "." in token and " " not in s_low:
        return token.rstrip("."), ""
    return "", ""


def _domain_matches(host: str, domain: str) -> bool:
    """Matches exact host or strict subdomain boundary ('.{domain}')."""
    if not host or not domain:
        return False
    return host == domain or host.endswith("." + domain)


def _pattern_matches_sender(pat: str, s_low: str, sender_host: str, sender_path: str) -> bool:
    """Boundary-aware matcher for EXPLICIT_AGENCY_PATTERNS."""
    if "@" in pat:
        return s_low == pat or re.search(r"(?<![a-z0-9._%+-])" + re.escape(pat) + r"(?![a-z0-9.-])", s_low) is not None
    if "/" in pat:
        dom_part, path_part = pat.split("/", 1)
        return _domain_matches(sender_host, dom_part) and sender_path.lstrip("/").startswith(path_part)
    if "." in pat:
        return _domain_matches(sender_host, pat)
    return re.search(r"(?<![a-z0-9])" + re.escape(pat) + r"(?![a-z0-9])", s_low) is not None


def feed_code_to_agency_and_country(feed_code):
    """Converts a canonical Alert Hub source_feed_id (e.g. 'sa-ncm-en', 'pw-wso', 'int-gdacs-en') into (source_feed_id, agency_id, country_code)."""
    if not feed_code:
        return None
    fc = str(feed_code).strip().lower()
    if fc == "google-public-alerts":
        return ("google-public-alerts", "GOOGLE_PUBLIC_ALERTS", "Google")
    parts = [p for p in re.split(r"[-_]", fc) if p]
    if len(parts) >= 2 and parts[0] == "int":
        abbrev = parts[1].upper()
        return (fc, f"INT_{abbrev}", "INT")
    if len(parts) >= 2 and len(parts[0]) == 2:
        iso2 = parts[0].upper()
        iso3 = ISO2_TO_ISO3.get(iso2, "Other")
        abbrev = parts[1].upper()
        return (fc, f"{iso2}_{abbrev}", iso3)
    return None


def resolve_feed_and_agency(sender, identifier=None, source_feed_id=None, area_desc=None, existing_country_code=None):
    """
    Resolves (source_feed_id, agency_id, country_code) deterministically using:
    1. Explicit transport-layer source_feed_id (ownerFeedCode / <cap-extensions:sourceFeed>) if provided.
    2. Registered authorEmail / agency domain / sender pattern.
    3. Custom agency <identifier> prefix (BMS_, IDEAM, AusBoM-, CWA-, TMD, IN-, KR., DOI-USGS, OnSolve-, etc.).
    4. WMO OID prefix (2.49.0.[01].<ISO_NUMERIC>.<AGENCY_SEQ>) or MeteoAlarm 2.49.0.3.0.<ISO2>.
    5. Sender ccTLD or area_desc fallback.
    """
    # 1. If transport-layer source_feed_id is explicitly provided, use it directly!
    if source_feed_id:
        parsed = feed_code_to_agency_and_country(source_feed_id)
        if parsed:
            return parsed

    s_raw = str(sender or "").strip()
    s_low = s_raw.lower()
    sender_host, sender_path = _extract_sender_host_and_path(s_low)
    id_raw = str(identifier or "").strip()
    area_low = str(area_desc or "").lower()
    cc = str(existing_country_code or "").strip()

    # 2. Check optional locally configured sender map (never embedded in public code)
    opt_map = _load_optional_sender_map()
    if s_low and s_low in opt_map:
        return opt_map[s_low]

    # 2b. Check explicit path / agency-token patterns (e.g. ntwc, ptwc)
    # before generic domain maps so specific sub-agencies are not shadowed.
    for pat, info in EXPLICIT_AGENCY_PATTERNS:
        if "/" in pat or "." not in pat:
            if _pattern_matches_sender(pat, s_low, sender_host, sender_path):
                return info

    # 3. Longest-domain-first boundary match across EXPLICIT_AGENCY_PATTERNS & FAH_DOMAIN_MAP
    if sender_host:
        domain_rules = []
        for pat, info in EXPLICIT_AGENCY_PATTERNS:
            if "." in pat and "@" not in pat and "/" not in pat:
                domain_rules.append((len(pat), 0, pat, info))
        for dom, (sid, ag_id, iso2) in FAH_DOMAIN_MAP.items():
            domain_rules.append((len(dom), 1, dom, (sid, ag_id, ISO2_TO_ISO3.get(iso2, iso2))))
        domain_rules.sort(key=lambda item: (-item[0], item[1]))
        for _, _, dom, info in domain_rules:
            if _domain_matches(sender_host, dom):
                return info

    # 4. China provincial/municipal meteorological & natural resources bureaus
    if any(
        k in s_low
        for k in [
            "meteorological_observatory",
            "meteorological_bureau",
            "meteorological_station",
            "natural_resources_bureau",
            "water_resources_bureau",
            "cn-cma",
            "_county",
            "_district",
            "_prefecture",
        ]
    ) or re.match(r"^\d{12,14}_\d{12,14}", id_raw):
        if "natural_resources" in s_low:
            return ("cn-mnr-zh", "CN_MNR", "CHN")
        return ("cn-cma-zh", "CN_CMA", "CHN")

    # 5. India IMD / CWC / SDMA / NDMA
    if s_raw == "CWC" or "central water commission" in s_low:
        return ("in-cwc-en", "IN_CWC", "IND")
    if s_raw.startswith("IMD-") or s_low.startswith("imd") or "india meteorological" in s_low:
        return ("in-imd-en", "IN_IMD", "IND")
    if "sdma" in s_low or "sndmc" in s_low or "ndma" in s_low or id_raw.startswith("IN-17"):
        return ("in-ndma-xx", "IN_NDMA", "IND")

    # 6. Identifier prefix & agency code rules
    if id_raw.startswith("BMS_"):
        return ("bb-bms-en", "BB_BMS", "BRB")
    if id_raw.startswith("IDEAM"):
        return ("co-ideam-es", "CO_IDEAM", "COL")
    if id_raw.startswith("AusBoM-"):
        return ("au-bom-en", "AU_BOM", "AUS")
    if id_raw.startswith(("CWA-", "CWB-")):
        return ("tw-cwa-zh", "TW_CWA", "TWN")
    if id_raw.startswith(
        (
            "TWC_",
            "Freeway_Alert_",
            "TP_",
            "NFA_",
            "QBTECH_",
            "TPMRT_",
            "TPC_",
            "TCMETRO_",
            "THB-",
            "HospitalReport_",
            "DONE_",
            "A202",
        )
    ) or any(k in s_low for k in [".taipei", ".tw", "taipei", "taipower", "digitw"]):
        return ("tw-ncdr-zh", "TW_NCDR", "TWN")
    if id_raw.startswith("TMD"):
        return ("th-tmd-en", "TH_TMD", "THA")
    if id_raw.startswith("KR.") or s_raw == "KMA":
        return ("kr-kma-weather-en", "KR_KMA", "KOR")
    if id_raw.startswith("DOI-USGS"):
        return ("us-usgs-volcano-en", "US_USGS", "USA")
    if s_raw == "LHMS":
        return ("lt-lhms-xx", "LT_LHMS", "LTU")
    if "philippine_red_cross" in s_low:
        return ("ph-prc-en", "PH_PRC", "PHL")
    if "new_zealand_red_cross" in s_low:
        return ("nz-redcross-en", "NZ_REDCROSS", "NZL")
    if "canadian_red_cross" in s_low:
        return ("ca-redcross-en", "CA_REDCROSS", "CAN")
    if "nemosvg" in s_low or s_raw in {"NEMO", "MinistryofHealthWellnessandtheEnvironment"}:
        return ("vc-meteo-en", "VC_METEO", "VCT")
    if s_raw in {"MeteorologicalServices", "DDM"}:
        return ("ag-ms-en", "AG_MS", "ATG")
    if "weather.cw" in s_low:
        return ("cw-meteo-en", "CW_METEO", "CUW")

    # 7. WMO OID prefix (2.49.0.[01].<country_num>.<agency_seq>)
    m = re.search(r"2\.49\.0\.[01]\.(\d+)\.(\d+)", id_raw + " " + s_raw)
    if m:
        c_num, a_seq = m.group(1), m.group(2)
        oid_key = f"{c_num}.{a_seq}"
        if oid_key in FAH_OID_MAP:
            sid, ag_id, iso2 = FAH_OID_MAP[oid_key]
            return (sid, ag_id, ISO2_TO_ISO3.get(iso2, iso2))
        if c_num in ISO_NUM_MAP:
            iso2, iso3 = ISO_NUM_MAP[c_num]
            if iso2 in FAH_ISO2_DEFAULT:
                sid, ag_id = FAH_ISO2_DEFAULT[iso2]
                return (sid, ag_id, iso3)
            return (f"{iso2.lower()}-wmo-{a_seq}", f"{iso2}_WMO_{a_seq}", iso3)

    # 8. Legacy MeteoAlarm 2.49.0.3.0.<ISO2>
    m_ma = re.search(r"2\.49\.0\.3\.0\.([A-Z]{2})\.", id_raw)
    if m_ma:
        iso2 = m_ma.group(1).upper()
        iso3 = ISO2_TO_ISO3.get(iso2, "Other")
        if iso2 in FAH_ISO2_DEFAULT:
            sid, ag_id = FAH_ISO2_DEFAULT[iso2]
            return (sid, ag_id, iso3)
        return (f"{iso2.lower()}-meteoalarm", f"{iso2}_METEOALARM", iso3)

    # 9. Canada NAAD / Alert Ready & US FEMA IPAWS municipal/state senders
    if any(
        k in s_low
        for k in [
            "sask",
            "manitoba",
            "alberta",
            "pelmorex",
            "alertready",
            "ontario",
            "bcemergency",
            "yukon",
            "provinceof",
            "northwestterritories",
            "calgary",
            "whitecourt",
            "provincialemergency",
            "grandeprairie",
            "barrhead",
            "rcmp",
            "townof",
            "villageof",
            "woodbuffalo",
        ]
    ):
        return ("ca-naad-en", "CA_NAAD", "CAN")

    if (
        id_raw.startswith(("OnSolve-", "AS-", "v2-", "CAPNET-", "GVP-"))
        or re.match(r"^20\d{4}_", s_raw)
        or re.match(r"^\d{15,25}$", id_raw)
        or _domain_matches(sender_host, "gov")
        or any(
            k in s_low
            for k in [
                "departmentofpublicsafety",
                "county",
                "sheriff",
                "nixle",
                "emnet-",
                "ashergroup",
                "state_police",
                "emergency_management_division",
                "si.edu",
                "iheartmedia",
                "kyeoc",
                "eas.localdomain",
            ]
        )
    ):
        return ("us-fema-ipaws-en", "US_FEMA_IPAWS", "USA")

    # 10. Sender ccTLD fallback
    tld = None
    if sender_host and "." in sender_host:
        cand_tld = sender_host.rsplit(".", 1)[-1]
        if len(cand_tld) == 2 and cand_tld.isalpha():
            tld = cand_tld.upper()
    if tld:
        if tld in ISO2_TO_ISO3 and tld not in {"CO", "IO", "AI", "ME", "TV", "FM"}:
            iso3 = ISO2_TO_ISO3[tld]
            if tld in FAH_ISO2_DEFAULT:
                sid, ag_id = FAH_ISO2_DEFAULT[tld]
                return (sid, ag_id, iso3)
            return (f"{tld.lower()}-nat-feed", f"{tld}_NAT_AGENCY", iso3)

    # 11. Existing country_code or area_desc fallback
    if cc and cc not in {"Other", "None", "NULL", "Google"} and cc in ISO3_TO_ISO2:
        iso2 = ISO3_TO_ISO2[cc]
        if iso2 in FAH_ISO2_DEFAULT:
            sid, ag_id = FAH_ISO2_DEFAULT[iso2]
            return (sid, ag_id, cc)
        return (f"{iso2.lower()}-nat-feed", f"{iso2}_NAT_AGENCY", cc)

    if "china" in area_low or any(ord(c) > 0x4E00 and ord(c) < 0x9FFF for c in s_raw):
        return ("cn-cma-zh", "CN_CMA", "CHN")
    if "colombia" in area_low:
        return ("co-ungrd-es", "CO_UNGRD", "COL")
    if "argentina" in area_low:
        return ("ar-smn-es", "AR_SMN", "ARG")

    # 12. Last-resort NAAD uppercase-UUID heuristic (only after ccTLD & existing_country_code)
    if re.match(r"^[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{6}", id_raw):
        return ("ca-naad-en", "CA_NAAD", "CAN")

    return ("unmapped-feed", "UNMAPPED_AGENCY", "Other")


def resolve_country_code(sender, identifier=None, area_desc=None, source_feed_id=None):
    """Backward-compatible helper returning the resolved 3-letter ISO3 country_code."""
    _, _, iso3 = resolve_feed_and_agency(
        sender, identifier=identifier, source_feed_id=source_feed_id, area_desc=area_desc
    )
    return iso3
