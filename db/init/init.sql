-- Create the uploaded_files table
CREATE TABLE IF NOT EXISTS uploaded_files (
    id SERIAL PRIMARY KEY,
    filename VARCHAR,
    content BYTEA,
    upload_datetime TIMESTAMP WITH TIME ZONE DEFAULT now(),
    category VARCHAR,
    CONSTRAINT _filename_category_uc UNIQUE (filename, category)
);

-- Change the owner of the table to the application user
ALTER TABLE uploaded_files OWNER TO "${POSTGRES_USER}";

-- Grant privileges to the application user
GRANT ALL PRIVILEGES ON TABLE uploaded_files TO "${POSTGRES_USER}";

-- Grant usage on the public schema
GRANT USAGE ON SCHEMA public TO "${POSTGRES_USER}";

-- Grant privileges on sequences and functions
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO "${POSTGRES_USER}";
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO "${POSTGRES_USER}";
