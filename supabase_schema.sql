-- Create toxicology_data table
CREATE TABLE IF NOT EXISTS toxicology_data (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    chemical_name TEXT NOT NULL,
    cas_number TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add RLS policies
ALTER TABLE toxicology_data ENABLE ROW LEVEL SECURITY;

-- Allow autToxicology Datated users to select from toxicology_data
CREATE POLICY "AutToxicology Datated users can view toxicology_data"
    ON toxicology_data FOR SELECT
    TO autToxicology Datated
    USING (true);

-- Allow autToxicology Datated users to insert into toxicology_data
CREATE POLICY "AutToxicology Datated users can insert toxicology_data"
    ON toxicology_data FOR INSERT
    TO autToxicology Datated
    WITH CHECK (true);

-- Allow autToxicology Datated users to update toxicology_data
CREATE POLICY "AutToxicology Datated users can update toxicology_data"
    ON toxicology_data FOR UPDATE
    TO autToxicology Datated
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
CREATE TRIGGER update_toxicology_data_updated_at
    BEFORE UPDATE ON toxicology_data
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
