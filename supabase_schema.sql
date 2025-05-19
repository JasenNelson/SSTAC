-- Create chemicals table
CREATE TABLE IF NOT EXISTS chemicals (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name TEXT NOT NULL,
    cas_number TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add RLS policies
ALTER TABLE chemicals ENABLE ROW LEVEL SECURITY;

-- Allow authenticated users to select from chemicals
CREATE POLICY "Authenticated users can view chemicals"
    ON chemicals FOR SELECT
    TO authenticated
    USING (true);

-- Allow authenticated users to insert into chemicals
CREATE POLICY "Authenticated users can insert chemicals"
    ON chemicals FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- Allow authenticated users to update chemicals
CREATE POLICY "Authenticated users can update chemicals"
    ON chemicals FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to update updated_at timestamp
CREATE TRIGGER update_chemicals_updated_at
    BEFORE UPDATE ON chemicals
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
