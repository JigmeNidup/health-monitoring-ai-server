import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database connection function
def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv('PSQL_HOST'),
        database=os.getenv('PSQL_DATABASE'),
        user=os.getenv('PSQL_USER'),
        password=os.getenv('PSQL_PASSWORD'),
        port=os.getenv('PSQL_PORT')
    )
    return conn