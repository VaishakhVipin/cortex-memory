# 🛠️ Auth System Implementation Guide

## 📋 **Quick Start Implementation**

### **1. Project Structure**
```
backend/
├── auth/
│   ├── __init__.py
│   ├── models.py          # Database models
│   ├── schemas.py         # Pydantic schemas
│   ├── services.py        # Business logic
│   ├── routes.py          # API endpoints
│   └── middleware.py      # Auth middleware
├── billing/
│   ├── __init__.py
│   ├── polar_client.py    # Polar.sh integration
│   ├── services.py        # Billing logic
│   └── webhooks.py        # Webhook handlers
├── core/
│   ├── config.py          # Settings
│   ├── database.py        # Database connection
│   └── security.py        # Security utilities
└── main.py                # FastAPI app
```

### **2. Core Dependencies**
```python
# requirements.txt
fastapi==0.104.1
sqlalchemy==2.0.23
alembic==1.12.1
psycopg2-binary==2.9.9
redis==5.0.1
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
httpx==0.25.2
pydantic==2.5.0
pydantic-settings==2.1.0
```

---

## 🔧 **Implementation Steps**

### **Step 1: Database Models**

#### **models.py**
```python
from sqlalchemy import Column, String, Boolean, DateTime, Integer, DECIMAL, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    company = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    email_verification_token = Column(String(255))
    two_factor_secret = Column(String(255))
    two_factor_enabled = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    subscription_tier = Column(String(50), default="free")
    polar_customer_id = Column(String(255))
    
    # Relationships
    api_keys = relationship("APIKey", back_populates="user")
    usage_logs = relationship("UsageLog", back_populates="user")

class APIKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    key_hash = Column(String(255), nullable=False)
    key_prefix = Column(String(8), nullable=False)
    name = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    permissions = Column(JSON, default={})
    rate_limit_per_minute = Column(Integer, default=100)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime)
    expires_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    usage_logs = relationship("UsageLog", back_populates="api_key")

class UsageLog(Base):
    __tablename__ = "usage_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    api_key_id = Column(UUID(as_uuid=True), ForeignKey("api_keys.id"), nullable=False)
    operation = Column(String(100), nullable=False)
    tokens_used = Column(Integer, default=0)
    cost_usd = Column(DECIMAL(10, 6), default=0)
    metadata = Column(JSON)
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="usage_logs")
    api_key = relationship("APIKey", back_populates="usage_logs")
```

### **Step 2: Pydantic Schemas**

#### **schemas.py**
```python
from pydantic import BaseModel, EmailStr, validator
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company: Optional[str] = None
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str
    two_factor_code: Optional[str] = None

class UserResponse(BaseModel):
    id: UUID
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    company: Optional[str]
    is_verified: bool
    subscription_tier: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class APIKeyCreate(BaseModel):
    name: str
    permissions: Dict[str, Any] = {}
    rate_limit_per_minute: Optional[int] = None

class APIKeyResponse(BaseModel):
    id: UUID
    name: str
    key_prefix: str
    is_active: bool
    permissions: Dict[str, Any]
    rate_limit_per_minute: int
    created_at: datetime
    last_used: Optional[datetime]
    
    class Config:
        from_attributes = True

class UsageStats(BaseModel):
    total_requests: int
    total_tokens: int
    total_cost_usd: float
    requests_today: int
    tokens_today: int
    cost_today_usd: float
```

### **Step 3: Security Utilities**

#### **security.py**
```python
import hashlib
import secrets
import bcrypt
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext

# Configuration
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class SecurityUtils:
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt."""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def generate_api_key() -> tuple[str, str, str]:
        """Generate API key and return (key, hash, prefix)."""
        key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        key_prefix = key[:8]
        return key, key_hash, key_prefix
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> Optional[dict]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError:
            return None
```

### **Step 4: Authentication Service**

#### **services.py**
```python
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import Optional, List
from datetime import datetime, timedelta
import secrets
import pyotp

from .models import User, APIKey, UsageLog
from .schemas import UserCreate, APIKeyCreate
from .security import SecurityUtils

class AuthService:
    def __init__(self, db: Session):
        self.db = db
    
    def create_user(self, user_data: UserCreate) -> User:
        """Create new user."""
        # Check if user already exists
        existing_user = self.db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise ValueError("User with this email already exists")
        
        # Hash password
        password_hash = SecurityUtils.hash_password(user_data.password)
        
        # Generate email verification token
        verification_token = secrets.token_urlsafe(32)
        
        # Create user
        user = User(
            email=user_data.email,
            password_hash=password_hash,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            company=user_data.company,
            email_verification_token=verification_token
        )
        
        try:
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            return user
        except IntegrityError:
            self.db.rollback()
            raise ValueError("User creation failed")
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password."""
        user = self.db.query(User).filter(User.email == email).first()
        if not user:
            return None
        
        if not SecurityUtils.verify_password(password, user.password_hash):
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        self.db.commit()
        
        return user
    
    def verify_email(self, token: str) -> bool:
        """Verify user email with token."""
        user = self.db.query(User).filter(
            User.email_verification_token == token
        ).first()
        
        if not user:
            return False
        
        user.is_verified = True
        user.email_verification_token = None
        self.db.commit()
        
        return True
    
    def create_api_key(self, user_id: str, key_data: APIKeyCreate) -> tuple[str, APIKey]:
        """Create new API key for user."""
        # Generate API key
        key, key_hash, key_prefix = SecurityUtils.generate_api_key()
        
        # Create API key record
        api_key = APIKey(
            user_id=user_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=key_data.name,
            permissions=key_data.permissions,
            rate_limit_per_minute=key_data.rate_limit_per_minute or 100
        )
        
        self.db.add(api_key)
        self.db.commit()
        self.db.refresh(api_key)
        
        return key, api_key
    
    def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """Validate API key and return associated record."""
        if len(api_key) < 8:
            return None
        
        key_prefix = api_key[:8]
        key_hash = SecurityUtils.hash_password(api_key)
        
        # Find API key by prefix first (for performance)
        db_key = self.db.query(APIKey).filter(
            APIKey.key_prefix == key_prefix,
            APIKey.is_active == True
        ).first()
        
        if not db_key:
            return None
        
        # Verify hash
        if db_key.key_hash != key_hash:
            return None
        
        # Update last used
        db_key.last_used = datetime.utcnow()
        self.db.commit()
        
        return db_key
    
    def get_user_api_keys(self, user_id: str) -> List[APIKey]:
        """Get all API keys for user."""
        return self.db.query(APIKey).filter(
            APIKey.user_id == user_id
        ).all()
    
    def deactivate_api_key(self, user_id: str, key_id: str) -> bool:
        """Deactivate API key."""
        api_key = self.db.query(APIKey).filter(
            APIKey.id == key_id,
            APIKey.user_id == user_id
        ).first()
        
        if not api_key:
            return False
        
        api_key.is_active = False
        self.db.commit()
        
        return True
```

### **Step 5: Polar.sh Integration**

#### **polar_client.py**
```python
import httpx
from typing import Dict, Any, Optional
from .config import settings

class PolarClient:
    def __init__(self):
        self.api_key = settings.POLAR_API_KEY
        self.base_url = "https://api.polar.sh/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def create_customer(self, email: str, name: str, metadata: Dict[str, Any] = None) -> str:
        """Create customer in Polar.sh."""
        customer_data = {
            "email": email,
            "name": name,
            "metadata": metadata or {}
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/customers",
                json=customer_data,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()["id"]
    
    async def create_subscription(self, customer_id: str, product_id: str) -> str:
        """Create subscription in Polar.sh."""
        subscription_data = {
            "customer_id": customer_id,
            "product_id": product_id,
            "status": "active"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/subscriptions",
                json=subscription_data,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()["id"]
    
    async def create_usage_charge(self, customer_id: str, amount_usd: float, description: str) -> Dict[str, Any]:
        """Create usage-based charge."""
        charge_data = {
            "customer_id": customer_id,
            "amount_usd": amount_usd,
            "description": description,
            "type": "usage"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/charges",
                json=charge_data,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
    
    async def get_customer(self, customer_id: str) -> Dict[str, Any]:
        """Get customer details."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/customers/{customer_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
```

### **Step 6: API Routes**

#### **routes.py**
```python
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import List

from .schemas import UserCreate, UserLogin, UserResponse, APIKeyCreate, APIKeyResponse
from .services import AuthService
from .security import SecurityUtils
from ..core.database import get_db
from ..core.config import settings

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()

@router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register new user."""
    auth_service = AuthService(db)
    
    try:
        user = auth_service.create_user(user_data)
        return UserResponse.from_orm(user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/login")
async def login(login_data: UserLogin, db: Session = Depends(get_db)):
    """Login user."""
    auth_service = AuthService(db)
    
    user = auth_service.authenticate_user(login_data.email, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Check 2FA if enabled
    if user.two_factor_enabled:
        if not login_data.two_factor_code:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="2FA code required"
            )
        
        totp = pyotp.TOTP(user.two_factor_secret)
        if not totp.verify(login_data.two_factor_code):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid 2FA code"
            )
    
    # Create access token
    access_token = SecurityUtils.create_access_token(
        data={"sub": str(user.id)}
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse.from_orm(user)
    }

@router.post("/verify-email")
async def verify_email(token: str, db: Session = Depends(get_db)):
    """Verify user email."""
    auth_service = AuthService(db)
    
    if auth_service.verify_email(token):
        return {"message": "Email verified successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification token"
        )

@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    key_data: APIKeyCreate,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Create new API key."""
    # Verify JWT token
    payload = SecurityUtils.verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    user_id = payload.get("sub")
    auth_service = AuthService(db)
    
    key, api_key = auth_service.create_api_key(user_id, key_data)
    
    return {
        "api_key": key,  # Only returned once!
        "api_key_info": APIKeyResponse.from_orm(api_key)
    }

@router.get("/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """List user's API keys."""
    payload = SecurityUtils.verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    user_id = payload.get("sub")
    auth_service = AuthService(db)
    
    api_keys = auth_service.get_user_api_keys(user_id)
    return [APIKeyResponse.from_orm(key) for key in api_keys]

@router.delete("/api-keys/{key_id}")
async def deactivate_api_key(
    key_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Deactivate API key."""
    payload = SecurityUtils.verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    user_id = payload.get("sub")
    auth_service = AuthService(db)
    
    if auth_service.deactivate_api_key(user_id, key_id):
        return {"message": "API key deactivated"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
```

### **Step 7: Middleware for API Key Authentication**

#### **middleware.py**
```python
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional
import time
import json

from .services import AuthService
from .models import UsageLog
from ..core.database import get_db
from ..core.redis_client import redis_client

class APIKeyMiddleware:
    def __init__(self):
        self.rate_limit_window = 60  # 1 minute
    
    async def __call__(self, request: Request, call_next):
        # Skip auth for certain endpoints
        if request.url.path in ["/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Get API key from header
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required"
            )
        
        # Validate API key
        db = next(get_db())
        auth_service = AuthService(db)
        api_key_record = auth_service.validate_api_key(api_key)
        
        if not api_key_record:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        # Check rate limiting
        if not await self.check_rate_limit(api_key_record):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Add user info to request state
        request.state.user_id = str(api_key_record.user_id)
        request.state.api_key_id = str(api_key_record.id)
        request.state.api_key_permissions = api_key_record.permissions
        
        # Process request
        response = await call_next(request)
        
        # Log usage
        await self.log_usage(request, response, api_key_record)
        
        return response
    
    async def check_rate_limit(self, api_key_record) -> bool:
        """Check if request is within rate limits."""
        key = f"rate_limit:{api_key_record.key_prefix}"
        
        # Get current count
        current = await redis_client.get(key)
        if current and int(current) >= api_key_record.rate_limit_per_minute:
            return False
        
        # Increment counter
        pipe = redis_client.pipeline()
        pipe.incr(key)
        pipe.expire(key, self.rate_limit_window)
        await pipe.execute()
        
        return True
    
    async def log_usage(self, request: Request, response, api_key_record):
        """Log API usage."""
        # Extract operation from path
        operation = request.url.path.replace("/", "_")
        
        # Calculate tokens used (simplified)
        tokens_used = len(request.body()) // 4  # Rough estimate
        
        # Calculate cost
        cost_per_token = 0.0001  # $0.0001 per token
        cost_usd = tokens_used * cost_per_token
        
        # Log to database
        db = next(get_db())
        usage_log = UsageLog(
            user_id=api_key_record.user_id,
            api_key_id=api_key_record.id,
            operation=operation,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent"),
            metadata={
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code
            }
        )
        
        db.add(usage_log)
        db.commit()
```

### **Step 8: Main Application**

#### **main.py**
```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .auth.routes import router as auth_router
from .auth.middleware import APIKeyMiddleware
from .billing.webhooks import router as billing_router
from .core.database import engine, Base
from .core.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    Base.metadata.create_all(bind=engine)
    yield
    # Shutdown
    pass

app = FastAPI(
    title="Cortex Memory Auth API",
    description="Authentication and billing API for Cortex Memory",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key middleware
app.add_middleware(APIKeyMiddleware)

# Include routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(billing_router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {
        "message": "Cortex Memory Auth API",
        "version": "1.0.0",
        "docs": "/docs"
    }
```

---

## 🚀 **Deployment**

### **1. Environment Variables**
```bash
# .env
DATABASE_URL=postgresql://user:password@localhost/cortex_auth
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key-here
POLAR_API_KEY=your-polar-api-key
ALLOWED_ORIGINS=["http://localhost:3000", "https://yourdomain.com"]
```

### **2. Docker Setup**
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **3. Docker Compose**
```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db/cortex_auth
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=cortex_auth
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

---

## 🧪 **Testing**

### **1. Unit Tests**
```python
# tests/test_auth.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from main import app
from auth.models import Base
from auth.services import AuthService

# Test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture
def db_session():
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client():
    return TestClient(app)

def test_user_registration(client, db_session):
    user_data = {
        "email": "test@example.com",
        "password": "testpassword123",
        "first_name": "Test",
        "last_name": "User"
    }
    
    response = client.post("/api/v1/auth/register", json=user_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["email"] == user_data["email"]
    assert data["is_verified"] == False

def test_api_key_creation(client, db_session):
    # First register user
    user_data = {
        "email": "test@example.com",
        "password": "testpassword123"
    }
    client.post("/api/v1/auth/register", json=user_data)
    
    # Login to get token
    login_data = {
        "email": "test@example.com",
        "password": "testpassword123"
    }
    login_response = client.post("/api/v1/auth/login", json=login_data)
    token = login_response.json()["access_token"]
    
    # Create API key
    key_data = {
        "name": "Test API Key",
        "permissions": {"read": True, "write": False}
    }
    
    response = client.post(
        "/api/v1/auth/api-keys",
        json=key_data,
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "api_key" in data
    assert data["api_key_info"]["name"] == "Test API Key"
```

---

## 📊 **Monitoring**

### **1. Key Metrics**
```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
api_requests_total = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method'])
api_request_duration = Histogram('api_request_duration_seconds', 'API request duration')
active_users = Gauge('active_users_total', 'Total active users')
api_keys_total = Gauge('api_keys_total', 'Total API keys')
usage_tokens_total = Counter('usage_tokens_total', 'Total tokens used')
usage_cost_total = Counter('usage_cost_total', 'Total cost in USD')
```

### **2. Health Checks**
```python
# monitoring/health.py
from sqlalchemy.orm import Session
from redis import Redis
import httpx

async def check_database(db: Session) -> bool:
    try:
        db.execute("SELECT 1")
        return True
    except:
        return False

async def check_redis(redis: Redis) -> bool:
    try:
        redis.ping()
        return True
    except:
        return False

async def check_polar_api() -> bool:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.polar.sh/health")
            return response.status_code == 200
    except:
        return False
```

---

**🎯 This implementation guide provides everything needed to build a production-ready authentication and billing system for Cortex Memory!** 