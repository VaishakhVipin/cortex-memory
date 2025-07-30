# 🔐 Backend Authentication System - Complete Specification

## 📋 **Table of Contents**

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Database Schema](#database-schema)
4. [API Endpoints](#api-endpoints)
5. [Authentication Flow](#authentication-flow)
6. [Polar.sh Integration](#polar-sh-integration)
7. [Rate Limiting & Usage Tracking](#rate-limiting--usage-tracking)
8. [Security Considerations](#security-considerations)
9. [Implementation Guide](#implementation-guide)
10. [Testing Strategy](#testing-strategy)

---

## 🎯 **Overview**

The Cortex Memory backend authentication system provides enterprise-grade user management, API key authentication, usage tracking, and billing integration with Polar.sh. This system enables the pay-per-use model for the Cortex Memory SDK.

### **Key Features**
- 🔑 **API Key Management**: Secure generation and validation
- 👤 **User Registration & Authentication**: Email/password + 2FA
- 📊 **Usage Tracking**: Real-time monitoring and analytics
- 💳 **Billing Integration**: Polar.sh payment processing
- 🚦 **Rate Limiting**: Tier-based access control
- 🔒 **Security**: JWT tokens, encryption, audit logs

---

## 🏗️ **Architecture**

### **System Components**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   Database      │
│   Dashboard     │◄──►│   (FastAPI)     │◄──►│   (PostgreSQL)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Polar.sh      │
                       │   Billing API   │
                       └─────────────────┘
```

### **Technology Stack**
- **Backend**: FastAPI + SQLAlchemy + Alembic
- **Database**: PostgreSQL + Redis (caching)
- **Authentication**: JWT + bcrypt
- **Payment**: Polar.sh API
- **Monitoring**: Prometheus + Grafana
- **Security**: HTTPS + CORS + Rate Limiting

---

## 🗄️ **Database Schema**

### **Users Table**
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    company VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    email_verification_token VARCHAR(255),
    two_factor_secret VARCHAR(255),
    two_factor_enabled BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP,
    subscription_tier VARCHAR(50) DEFAULT 'free',
    polar_customer_id VARCHAR(255)
);
```

### **API Keys Table**
```sql
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) NOT NULL,
    key_prefix VARCHAR(8) NOT NULL,
    name VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    permissions JSONB DEFAULT '{}',
    rate_limit_per_minute INTEGER DEFAULT 100,
    created_at TIMESTAMP DEFAULT NOW(),
    last_used TIMESTAMP,
    expires_at TIMESTAMP
);
```

### **Usage Logs Table**
```sql
CREATE TABLE usage_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    api_key_id UUID REFERENCES api_keys(id),
    operation VARCHAR(100) NOT NULL,
    tokens_used INTEGER DEFAULT 0,
    cost_usd DECIMAL(10,6) DEFAULT 0,
    metadata JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### **Billing Events Table**
```sql
CREATE TABLE billing_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    polar_event_id VARCHAR(255),
    event_type VARCHAR(100) NOT NULL,
    amount_usd DECIMAL(10,2),
    currency VARCHAR(3) DEFAULT 'USD',
    status VARCHAR(50) DEFAULT 'pending',
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### **Subscription Plans Table**
```sql
CREATE TABLE subscription_plans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    polar_product_id VARCHAR(255),
    price_usd DECIMAL(10,2) NOT NULL,
    tokens_per_month INTEGER,
    rate_limit_per_minute INTEGER,
    features JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 🔌 **API Endpoints**

### **Authentication Endpoints**

#### **POST /auth/register**
```json
{
    "email": "user@example.com",
    "password": "secure_password",
    "first_name": "John",
    "last_name": "Doe",
    "company": "Acme Corp"
}
```

#### **POST /auth/login**
```json
{
    "email": "user@example.com",
    "password": "secure_password",
    "two_factor_code": "123456" // Optional
}
```

#### **POST /auth/verify-email**
```json
{
    "token": "email_verification_token"
}
```

#### **POST /auth/forgot-password**
```json
{
    "email": "user@example.com"
}
```

#### **POST /auth/reset-password**
```json
{
    "token": "reset_token",
    "new_password": "new_secure_password"
}
```

### **API Key Management**

#### **POST /api-keys/create**
```json
{
    "name": "Production API Key",
    "permissions": {
        "context_generation": true,
        "analytics": true,
        "drift_detection": false
    },
    "rate_limit_per_minute": 1000
}
```

#### **GET /api-keys/list**
Returns all API keys for the authenticated user.

#### **DELETE /api-keys/{key_id}**
Deactivates an API key.

### **Usage & Billing**

#### **GET /usage/current**
Returns current usage statistics.

#### **GET /usage/history**
```json
{
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "group_by": "day"
}
```

#### **GET /billing/invoices**
Returns billing history and invoices.

---

## 🔄 **Authentication Flow**

### **1. User Registration**
```
1. User submits registration form
2. Validate email format and password strength
3. Hash password with bcrypt
4. Generate email verification token
5. Send verification email
6. Create user record (unverified)
7. Return success response
```

### **2. Email Verification**
```
1. User clicks verification link
2. Validate verification token
3. Mark user as verified
4. Create free tier subscription
5. Generate welcome API key
6. Send welcome email
```

### **3. API Key Authentication**
```
1. Client sends request with API key
2. Extract key prefix for fast lookup
3. Find API key in database
4. Validate key hash
5. Check rate limits
6. Log usage
7. Process request
```

### **4. Usage Tracking**
```
1. Track operation type and tokens used
2. Calculate cost based on tier
3. Update usage counters
4. Log to usage_logs table
5. Check billing thresholds
6. Trigger billing events if needed
```

---

## 💳 **Polar.sh Integration**

### **Polar.sh Setup**

#### **1. Product Configuration**
```python
POLAR_PRODUCTS = {
    "free": {
        "product_id": "prod_free_tier",
        "price_usd": 0.00,
        "tokens_per_month": 1000,
        "rate_limit": 10
    },
    "starter": {
        "product_id": "prod_starter_tier", 
        "price_usd": 29.00,
        "tokens_per_month": 100000,
        "rate_limit": 100
    },
    "professional": {
        "product_id": "prod_professional_tier",
        "price_usd": 99.00,
        "tokens_per_month": 1000000,
        "rate_limit": 1000
    },
    "enterprise": {
        "product_id": "prod_enterprise_tier",
        "price_usd": 299.00,
        "tokens_per_month": 10000000,
        "rate_limit": 10000
    }
}
```

#### **2. Webhook Handlers**
```python
@router.post("/webhooks/polar")
async def handle_polar_webhook(request: Request):
    """Handle Polar.sh webhooks for subscription events."""
    
    # Verify webhook signature
    signature = request.headers.get("X-Polar-Signature")
    if not verify_polar_signature(request.body(), signature):
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    event_data = await request.json()
    event_type = event_data.get("type")
    
    if event_type == "subscription.created":
        await handle_subscription_created(event_data)
    elif event_type == "subscription.updated":
        await handle_subscription_updated(event_data)
    elif event_type == "subscription.cancelled":
        await handle_subscription_cancelled(event_data)
    elif event_type == "invoice.paid":
        await handle_invoice_paid(event_data)
    
    return {"status": "success"}
```

#### **3. Billing Integration**
```python
class PolarBillingService:
    def __init__(self):
        self.api_key = settings.POLAR_API_KEY
        self.base_url = "https://api.polar.sh/api/v1"
    
    async def create_customer(self, user: User) -> str:
        """Create customer in Polar.sh."""
        customer_data = {
            "email": user.email,
            "name": f"{user.first_name} {user.last_name}",
            "metadata": {"user_id": str(user.id)}
        }
        
        response = await self._make_request("POST", "/customers", customer_data)
        return response["id"]
    
    async def create_subscription(self, customer_id: str, product_id: str) -> str:
        """Create subscription in Polar.sh."""
        subscription_data = {
            "customer_id": customer_id,
            "product_id": product_id,
            "status": "active"
        }
        
        response = await self._make_request("POST", "/subscriptions", subscription_data)
        return response["id"]
    
    async def create_usage_based_charge(self, customer_id: str, amount_usd: float, description: str):
        """Create usage-based charge."""
        charge_data = {
            "customer_id": customer_id,
            "amount_usd": amount_usd,
            "description": description,
            "type": "usage"
        }
        
        return await self._make_request("POST", "/charges", charge_data)
```

---

## 🚦 **Rate Limiting & Usage Tracking**

### **Rate Limiting Strategy**

#### **1. Tier-Based Limits**
```python
RATE_LIMITS = {
    "free": {
        "requests_per_minute": 10,
        "requests_per_hour": 100,
        "requests_per_day": 1000
    },
    "starter": {
        "requests_per_minute": 100,
        "requests_per_hour": 1000,
        "requests_per_day": 10000
    },
    "professional": {
        "requests_per_minute": 1000,
        "requests_per_hour": 10000,
        "requests_per_day": 100000
    },
    "enterprise": {
        "requests_per_minute": 10000,
        "requests_per_hour": 100000,
        "requests_per_day": 1000000
    }
}
```

#### **2. Redis-Based Rate Limiting**
```python
class RateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def check_rate_limit(self, api_key: str, operation: str) -> bool:
        """Check if request is within rate limits."""
        key = f"rate_limit:{api_key}:{operation}"
        
        # Get current count
        current = await self.redis.get(key)
        if current and int(current) >= self.get_limit(api_key):
            return False
        
        # Increment counter
        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 60)  # 1 minute window
        await pipe.execute()
        
        return True
```

### **Usage Tracking**

#### **1. Token Counting**
```python
class UsageTracker:
    def __init__(self, db_session, polar_service):
        self.db = db_session
        self.polar = polar_service
    
    async def track_usage(self, user_id: str, api_key_id: str, operation: str, 
                         tokens_used: int, metadata: dict = None):
        """Track API usage and calculate costs."""
        
        # Calculate cost based on tier
        user = await self.get_user(user_id)
        cost_per_token = self.get_cost_per_token(user.subscription_tier)
        cost_usd = tokens_used * cost_per_token
        
        # Log usage
        usage_log = UsageLog(
            user_id=user_id,
            api_key_id=api_key_id,
            operation=operation,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            metadata=metadata
        )
        
        self.db.add(usage_log)
        await self.db.commit()
        
        # Check if billing threshold reached
        monthly_usage = await self.get_monthly_usage(user_id)
        if monthly_usage > user.monthly_token_limit:
            await self.trigger_overage_billing(user_id, monthly_usage)
```

---

## 🔒 **Security Considerations**

### **1. API Key Security**
- **Hashing**: Store only SHA-256 hashes of API keys
- **Prefix Lookup**: Use first 8 characters for fast database lookup
- **Rotation**: Support API key rotation without downtime
- **Expiration**: Automatic expiration of unused keys

### **2. Authentication Security**
- **Password Hashing**: bcrypt with salt rounds
- **JWT Tokens**: Short-lived access tokens with refresh tokens
- **2FA**: TOTP-based two-factor authentication
- **Rate Limiting**: Prevent brute force attacks

### **3. Data Protection**
- **Encryption**: Encrypt sensitive data at rest
- **Audit Logs**: Complete audit trail of all operations
- **GDPR Compliance**: Data retention and deletion policies
- **PCI Compliance**: Secure handling of billing data

### **4. API Security**
- **HTTPS Only**: All endpoints require HTTPS
- **CORS**: Proper CORS configuration
- **Input Validation**: Comprehensive input sanitization
- **SQL Injection**: Parameterized queries only

---

## 🛠️ **Implementation Guide**

### **Phase 1: Core Authentication (Week 1-2)**

#### **1. Database Setup**
```bash
# Create database migrations
alembic init alembic
alembic revision --autogenerate -m "Initial auth schema"
alembic upgrade head
```

#### **2. User Management**
```python
# models/user.py
class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    # ... other fields
```

#### **3. API Key Management**
```python
# services/api_key_service.py
class APIKeyService:
    def generate_api_key(self) -> tuple[str, str]:
        """Generate new API key and return (key, hash)."""
        key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return key, key_hash
```

### **Phase 2: Polar.sh Integration (Week 3-4)**

#### **1. Webhook Setup**
```python
# routes/webhooks.py
@router.post("/polar")
async def polar_webhook(request: Request):
    # Handle subscription events
    pass
```

#### **2. Billing Service**
```python
# services/billing_service.py
class BillingService:
    def __init__(self, polar_client):
        self.polar = polar_client
    
    async def create_subscription(self, user_id: str, plan: str):
        # Create subscription in Polar.sh
        pass
```

### **Phase 3: Usage Tracking (Week 5-6)**

#### **1. Rate Limiting**
```python
# middleware/rate_limiting.py
@router.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Implement rate limiting logic
    pass
```

#### **2. Usage Analytics**
```python
# services/analytics_service.py
class AnalyticsService:
    async def get_usage_stats(self, user_id: str, period: str):
        # Generate usage reports
        pass
```

---

## 🧪 **Testing Strategy**

### **1. Unit Tests**
```python
# tests/test_auth.py
class TestAuthentication:
    async def test_user_registration(self):
        # Test user registration flow
        pass
    
    async def test_api_key_validation(self):
        # Test API key validation
        pass
```

### **2. Integration Tests**
```python
# tests/test_polar_integration.py
class TestPolarIntegration:
    async def test_subscription_creation(self):
        # Test Polar.sh subscription flow
        pass
```

### **3. Load Testing**
```python
# tests/load_test.py
async def test_rate_limiting():
    # Test rate limiting under load
    pass
```

### **4. Security Testing**
```python
# tests/security_test.py
class TestSecurity:
    async def test_sql_injection_prevention(self):
        # Test SQL injection prevention
        pass
    
    async def test_jwt_token_security(self):
        # Test JWT token security
        pass
```

---

## 📊 **Monitoring & Analytics**

### **1. Key Metrics**
- **User Registration Rate**: New users per day
- **API Usage**: Requests per minute/hour/day
- **Revenue**: Monthly recurring revenue (MRR)
- **Churn Rate**: User retention metrics
- **Error Rates**: API error percentages

### **2. Alerting**
- **High Error Rate**: >5% error rate
- **Rate Limit Exceeded**: Multiple users hitting limits
- **Billing Failures**: Failed payment processing
- **System Downtime**: Service availability

### **3. Dashboards**
- **User Dashboard**: Personal usage and billing
- **Admin Dashboard**: System-wide metrics
- **Billing Dashboard**: Revenue and subscription data

---

## 🚀 **Deployment Checklist**

### **Pre-Deployment**
- [ ] Database migrations tested
- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Polar.sh webhook endpoints configured
- [ ] Rate limiting tested
- [ ] Security audit completed

### **Post-Deployment**
- [ ] Monitor error rates
- [ ] Verify webhook delivery
- [ ] Test billing flows
- [ ] Check rate limiting
- [ ] Validate audit logs
- [ ] Performance monitoring

---

## 📚 **Additional Resources**

### **Documentation**
- [FastAPI Authentication](https://fastapi.tiangolo.com/tutorial/security/)
- [Polar.sh API Documentation](https://docs.polar.sh/)
- [PostgreSQL Best Practices](https://www.postgresql.org/docs/current/)
- [Redis Rate Limiting](https://redis.io/topics/patterns/distributed-locks)

### **Security Guidelines**
- [OWASP API Security](https://owasp.org/www-project-api-security/)
- [JWT Best Practices](https://auth0.com/blog/a-look-at-the-latest-draft-for-jwt-bcp/)
- [PCI DSS Requirements](https://www.pcisecuritystandards.org/)

---

**🎯 This specification provides a complete foundation for implementing enterprise-grade authentication and billing for Cortex Memory!** 