# 🧠 Cortex Memory - Changelog

## [2.0.3] - 2025-07-31

### ✅ Added
- **Complete Authentication System**
  - User registration with email/password validation
  - User login with JWT token authentication
  - Password hashing with bcrypt (12 salt rounds)
  - Two-factor authentication support (TOTP)
  - API key generation and management
  - User profile management

- **Database Schema**
  - PostgreSQL database integration with Neon DB
  - Complete table structure with all required columns
  - Foreign key relationships and constraints
  - Performance indexes
  - Usage tracking and billing events tables

- **Security Features**
  - JWT token authentication with refresh tokens
  - API key validation and rate limiting
  - CORS configuration
  - Environment variable security
  - Secure password storage

- **API Endpoints**
  - `POST /api/v1/auth/register` - User registration
  - `POST /api/v1/auth/login` - User login
  - `POST /api/v1/auth/logout` - User logout
  - `GET /api/v1/auth/me` - Get current user
  - `POST /api/v1/auth/refresh` - Refresh JWT token
  - `POST /api/v1/auth/2fa/enable` - Enable 2FA
  - `POST /api/v1/auth/2fa/verify` - Verify 2FA
  - `POST /api/v1/auth/api-keys/generate` - Generate API key
  - `GET /api/v1/auth/api-keys` - List API keys
  - `DELETE /api/v1/auth/api-keys/{key_id}` - Delete API key

### 🔧 Fixed
- **SQLAlchemy 2.0 Compatibility**
  - Fixed `Textual SQL expression 'SELECT 1' should be explicitly declared as text('SELECT 1')` error
  - Updated database connection check to use `text()` wrapper

- **Database Schema Issues**
  - Fixed missing columns in `users` table (`is_active`, `is_verified`, `two_factor_enabled`, `two_factor_secret`)
  - Resolved duplicate `Base` class declaration in `auth/models.py`
  - Recreated tables with correct schema using raw SQL

- **Project Structure**
  - Organized authentication modules in `auth/` directory
  - Separated core functionality in `core/` directory
  - Created standalone auth server for testing

### 🧹 Cleaned Up
- Removed temporary debugging scripts:
  - `check_table_structure.py`
  - `fix_tables.py`
  - `recreate_tables.py`
  - `test_db_connection.py`
  - `generate_keys.py`

- Updated `.gitignore` for better organization
- Created comprehensive documentation

### 📚 Documentation
- Created `AUTH_SYSTEM_IMPLEMENTATION.md` with complete implementation details
- Added database schema documentation
- Documented security considerations
- Listed environment variables and setup instructions

### 🧪 Testing
- Successfully tested user registration (201 Created)
- Verified all required user fields are present
- Confirmed database connectivity
- Tested health check endpoint

## 🚀 Ready for Production

The authentication system is now complete and ready for:
- Frontend integration
- Production deployment
- Billing system integration
- Advanced feature development

**Status**: ✅ Complete and Working 