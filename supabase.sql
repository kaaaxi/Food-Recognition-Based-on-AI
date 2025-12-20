-- users 
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    name TEXT,
    salt TEXT,
    password_hash TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- analysis_history
CREATE TABLE IF NOT EXISTS analysis_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT,
    dish_name TEXT,
    calories NUMERIC,
    protein NUMERIC,
    fat NUMERIC,
    carbs NUMERIC,
    portion_grams NUMERIC,
    confidence NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    raw_payload JSONB
);

-- Row Level Security
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_history ENABLE ROW LEVEL SECURITY;

-- anonymous access
CREATE POLICY "Allow all for users" ON users FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all for analysis_history" ON analysis_history FOR ALL USING (true) WITH CHECK (true);