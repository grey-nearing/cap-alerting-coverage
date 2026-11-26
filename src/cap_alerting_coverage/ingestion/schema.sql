-- Portable SQL DDL Schema for CAP Alert & Disaster Event Ingestion
-- Compatible with PostgreSQL and SQLite (adapt TIMESTAMP/BOOLEAN types as needed).

CREATE TABLE IF NOT EXISTS cap_alerts (
    id VARCHAR(36) PRIMARY KEY,
    source_site TEXT NOT NULL,
    identifier TEXT NOT NULL,
    sender TEXT,
    sent_at TIMESTAMP,
    status TEXT,
    msg_type TEXT,
    raw_xml TEXT,
    created_at TIMESTAMP,
    event TEXT,
    severity TEXT,
    urgency TEXT,
    certainty TEXT,
    scope TEXT,
    category TEXT,
    headline TEXT,
    description TEXT,
    instruction TEXT,
    area_desc TEXT,
    polygon TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    country_code VARCHAR(16),
    source_language VARCHAR(16),
    translated_headline TEXT,
    translated_description TEXT,
    translated_instruction TEXT,
    is_from_cap BOOLEAN DEFAULT TRUE,
    is_from_filtered_hub BOOLEAN DEFAULT FALSE,
    is_from_ifrc BOOLEAN DEFAULT FALSE,
    is_from_wmo_swic BOOLEAN DEFAULT FALSE,
    is_from_google BOOLEAN DEFAULT FALSE,
    source_feed_id TEXT,
    agency_id TEXT,
    source_url TEXT
);

CREATE INDEX IF NOT EXISTS idx_cap_alerts_sent_at ON cap_alerts (sent_at);
CREATE INDEX IF NOT EXISTS idx_cap_alerts_country ON cap_alerts (country_code);
CREATE INDEX IF NOT EXISTS idx_cap_alerts_event ON cap_alerts (event);
CREATE INDEX IF NOT EXISTS idx_cap_alerts_agency ON cap_alerts (agency_id);
CREATE INDEX IF NOT EXISTS idx_cap_alerts_source_feed ON cap_alerts (source_feed_id);

CREATE TABLE IF NOT EXISTS cap_source_feeds (
    source_feed_id TEXT PRIMARY KEY,
    agency_id TEXT NOT NULL,
    country_iso2 VARCHAR(8),
    country_code VARCHAR(16),
    authority_abbrev TEXT,
    agency_name TEXT,
    agency_name_native TEXT,
    languages TEXT,
    wmo_guid TEXT,
    wmo_register_url TEXT,
    cap_alert_feed_url TEXT,
    cap_alert_feed_status TEXT,
    source_is_official BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cap_source_feeds_country ON cap_source_feeds (country_code);

CREATE TABLE IF NOT EXISTS idmc_events (
    event_id BIGINT PRIMARY KEY,
    iso3 VARCHAR(8) NOT NULL,
    country TEXT NOT NULL,
    hazard_type TEXT NOT NULL,
    event_name TEXT,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    displacements BIGINT DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_idmc_events_iso3_start ON idmc_events (iso3, start_date);

CREATE TABLE IF NOT EXISTS ingestion_metrics (
    metric_key TEXT PRIMARY KEY,
    current_value BIGINT,
    target_value BIGINT,
    updated_at TIMESTAMP
);
