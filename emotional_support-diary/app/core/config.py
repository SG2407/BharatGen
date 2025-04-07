from pydantic import BaseSettings

class Settings(BaseSettings):
    MONGODB_URL: str = "your_mongodb_url"  # Optional, for database integration
    JWT_SECRET_KEY: str = "your_jwt_secret_key"  # Optional, for authentication

    class Config:
        env_file = ".env"  # You can keep this if you plan to use a .env file for other variables

settings = Settings()
