# 🧠 Authentication System

## Overview

The Cortex Memory authentication system provides secure user management, API key handling, and usage tracking for the Cortex Memory backend.

## 🔒 Security Features

- **Password Hashing**: bcrypt with configurable rounds
- **JWT Tokens**: Secure token-based authentication
- **API Key Management**: Secure API key generation and validation
- **Two-Factor Authentication**: TOTP-based 2FA with backup codes
- **Rate Limiting**: Built-in rate limiting for API endpoints
- **Usage Tracking**: Comprehensive usage analytics and billing

## 🏗️ Architecture

### Models
- `User`: User accounts with email, password hash, subscription tier
- `APIKey`: API key management with user relationships
- `UsageLog`: Usage tracking for billing and analytics
- `BillingEvent`: Polar.sh billing event tracking

### Services
- `AuthService`: Core business logic for authentication
- `SecurityUtils`: Security utilities and helper functions

### Routes
- User registration and authentication
- API key management
- Usage statistics and billing
- Two-factor authentication
- Password management

## 🔧 Setup

1. **Environment Variables**: Copy `env.example` to `.env` and configure:
   ```bash
   cp env.example .env
   ```

2. **Required Environment Variables**:
   - `POSTGRES_URL`: Database connection string
   - `JWT_SECRET_KEY`: Strong secret for JWT tokens
   - `SECRET_KEY`: Application secret key

3. **Database Setup**: Run the database connection test:
   ```bash
   python test_db_connection.py
   ```

## 🚀 Usage

### User Registration
```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePassword123!",
    "confirm_password": "SecurePassword123!"
  }'
```

### User Login
```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePassword123!"
  }'
```

### Create API Key
```bash
curl -X POST "http://localhost:8000/api/v1/auth/api-keys" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My API Key"
  }'
```

### Use API Key
```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "prompt": "Hello, how are you?",
    "provider": "gemini"
  }'
```

## 🔐 Security Considerations

### What's Secure
- ✅ Passwords are hashed with bcrypt
- ✅ API keys are hashed before storage
- ✅ JWT tokens use strong secrets
- ✅ Environment variables for sensitive data
- ✅ Rate limiting on all endpoints
- ✅ Input validation and sanitization

### What to Secure in Production
- 🔒 Use strong, unique secrets for JWT_SECRET_KEY
- 🔒 Enable HTTPS in production
- 🔒 Configure proper CORS origins
- 🔒 Use a secrets management service
- 🔒 Enable database SSL connections
- 🔒 Set up proper logging and monitoring
- 🔒 Implement email verification
- 🔒 Set up proper backup and recovery

### Neon Database SDK
Neon provides a Python SDK for easier database management:

```bash
pip install neon-python
```

```python
from neon import Neon

# Initialize client
neon = Neon(api_key="your-neon-api-key")

# Create database
database = neon.create_database("cortex-memory-prod")

# Get connection string
connection_string = database.connection_string
```

## 📊 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `GET /api/v1/auth/me` - Get current user info
- `PUT /api/v1/auth/me` - Update user info

### API Keys
- `POST /api/v1/auth/api-keys` - Create API key
- `GET /api/v1/auth/api-keys` - List API keys
- `DELETE /api/v1/auth/api-keys/{id}` - Deactivate API key

### Usage & Billing
- `GET /api/v1/auth/usage/stats` - Get usage statistics
- `GET /api/v1/auth/billing/info` - Get billing information

### Two-Factor Authentication
- `POST /api/v1/auth/2fa/setup` - Setup 2FA
- `POST /api/v1/auth/2fa/enable` - Enable 2FA
- `POST /api/v1/auth/2fa/verify` - Verify 2FA code
- `POST /api/v1/auth/2fa/disable` - Disable 2FA

### Password Management
- `POST /api/v1/auth/password/change` - Change password
- `POST /api/v1/auth/password/reset` - Request password reset
- `POST /api/v1/auth/password/reset/confirm` - Confirm password reset

## 🧪 Testing

Run the authentication system tests:

```bash
# Test database connection
python test_db_connection.py

# Test the full system
python -m pytest tests/test_auth.py
```

## 📚 Documentation

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Neon Database Documentation](https://neon.tech/docs)
- [Polar.sh Documentation](https://polar.sh/docs) 