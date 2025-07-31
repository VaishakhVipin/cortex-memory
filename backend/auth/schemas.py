"""
🧠 Authentication Schemas
Pydantic models for request/response validation
"""

from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List
from datetime import datetime
from uuid import UUID

# Base schemas
class UserBase(BaseModel):
    email: EmailStr
    
class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=128)
    confirm_password: str = Field(..., min_length=8, max_length=128)
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(UserBase):
    id: UUID
    subscription_tier: str
    is_active: bool
    is_verified: bool
    two_factor_enabled: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    subscription_tier: Optional[str] = None
    is_active: Optional[bool] = None

# API Key schemas
class APIKeyBase(BaseModel):
    name: Optional[str] = Field(None, max_length=255)

class APIKeyCreate(APIKeyBase):
    pass

class APIKeyResponse(APIKeyBase):
    id: UUID
    user_id: UUID
    is_active: bool
    created_at: datetime
    last_used_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class APIKeyList(BaseModel):
    api_keys: List[APIKeyResponse]
    total: int

# Usage schemas
class UsageLogResponse(BaseModel):
    id: UUID
    user_id: UUID
    api_key_id: Optional[UUID]
    endpoint: str
    tokens_used: int
    cost_usd: float
    provider: Optional[str]
    response_time_ms: Optional[int]
    status_code: Optional[int]
    created_at: datetime
    
    class Config:
        from_attributes = True

class UsageStats(BaseModel):
    total_requests: int
    total_tokens: int
    total_cost_usd: float
    requests_today: int
    tokens_today: int
    cost_today_usd: float
    requests_this_month: int
    tokens_this_month: int
    cost_this_month_usd: float

# Token schemas
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

class TokenData(BaseModel):
    user_id: Optional[UUID] = None
    email: Optional[str] = None

# 2FA schemas
class TwoFactorSetup(BaseModel):
    secret: str
    qr_code_url: str
    backup_codes: List[str]

class TwoFactorVerify(BaseModel):
    code: str = Field(..., min_length=6, max_length=6)
    
    @validator('code')
    def validate_code(cls, v):
        if not v.isdigit():
            raise ValueError('Code must contain only digits')
        return v

class TwoFactorDisable(BaseModel):
    password: str

# Password schemas
class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)
    confirm_new_password: str = Field(..., min_length=8, max_length=128)
    
    @validator('confirm_new_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('New passwords do not match')
        return v

class PasswordReset(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8, max_length=128)
    confirm_new_password: str = Field(..., min_length=8, max_length=128)
    
    @validator('confirm_new_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('New passwords do not match')
        return v

# Response schemas
class MessageResponse(BaseModel):
    message: str
    success: bool = True

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None

# Rate limiting schemas
class RateLimitInfo(BaseModel):
    limit: int
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None

# Billing schemas
class BillingInfo(BaseModel):
    subscription_tier: str
    monthly_limit: int
    current_usage: int
    remaining_usage: int
    next_billing_date: Optional[datetime] = None
    total_cost_this_month: float

class SubscriptionTier(BaseModel):
    name: str
    price_usd: float
    monthly_limit: int
    features: List[str]
    is_current: bool = False 