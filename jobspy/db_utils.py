import json
import os
from sqlalchemy import create_engine, text

# find the config in current directory
config_filepath = os.path.join(os.getcwd(), 'config.json')
if not os.path.exists(config_filepath):
    with open(config_filepath, 'w') as f:
        config_template = {
            "SERVER_NAME": "DESKTOP-JJT739B\\SQLEXPRESS",
            "DATABASE_NAME": "jobs",
            "DRIVER_NAME": "ODBC Driver 17 for SQL Server",
            "DB_CONNECTION_URI": 'mssql+pyodbc://{SERVER_NAME}/{DATABASE_NAME}?driver={DRIVER_NAME}'
        }
        json.dump(config_template, f, indent=4)
        raise FileNotFoundError("config.json file not found. A template has been created. Please fill in the database connection details.")


with open(config_filepath, 'r') as f:
    config = json.load(f)


SERVER_NAME = config.get("SERVER_NAME")
DATABASE_NAME = config.get("DATABASE_NAME")
DRIVER_NAME = config.get("DRIVER_NAME")
DB_CONNECTION_URI = config.get("DB_CONNECTION_URI").format(
    SERVER_NAME=SERVER_NAME,
    DATABASE_NAME=DATABASE_NAME,
    DRIVER_NAME=DRIVER_NAME
)


engine = create_engine(DB_CONNECTION_URI)


def create_required_tables():
    with engine.connect() as connection:
        connection.execute(text("""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='scraped_jobs' AND xtype='U')
        CREATE TABLE scraped_jobs (
            id INT PRIMARY KEY IDENTITY(1,1),
            site NVARCHAR(50),
            job_url NVARCHAR(500),
            job_url_direct NVARCHAR(500),
            title NVARCHAR(300),
            company NVARCHAR(200),
            location NVARCHAR(200),
            date_posted DATETIME,
            job_type NVARCHAR(100),
            salary_source NVARCHAR(100),
            interval NVARCHAR(50),
            min_amount FLOAT,
            max_amount FLOAT,
            currency NVARCHAR(10),
            is_remote BIT,
            job_level NVARCHAR(100),
            job_function NVARCHAR(200),
            listing_type NVARCHAR(100),
            emails NVARCHAR(MAX),
            description NVARCHAR(MAX),
            company_industry NVARCHAR(200),
            company_url NVARCHAR(500),
            company_logo NVARCHAR(500),
            company_url_direct NVARCHAR(500),
            company_addresses NVARCHAR(MAX),
            company_num_employees NVARCHAR(100),
            company_revenue NVARCHAR(100),
            company_description NVARCHAR(MAX),
            skills NVARCHAR(MAX),
            experience_range NVARCHAR(100),
            company_rating FLOAT,
            company_reviews_count INT,
            vacancy_count INT,
            work_from_home_type NVARCHAR(100)
        )
        """))

    connection.commit()

def insert_scraped_jobs(jobs_df):
    with engine.connect() as connection:
        jobs_df.to_sql('scraped_jobs', con=connection, if_exists='append', index=False)