import os
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))  # default fallback
REDIS_USERNAME = os.getenv("REDIS_USERNAME")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Add REDIS_CONFIG for redis_client.py compatibility
REDIS_CONFIG = {
    "host": REDIS_HOST,
    "port": REDIS_PORT,
    "decode_responses": True,
    "username": REDIS_USERNAME,
    "password": REDIS_PASSWORD,
}
