"""
🧠 Authentication Routes
FastAPI routes for user authentication and API key management
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from datetime import timedelta, datetime

from .models import User
from .schemas import (
    UserCreate, UserLogin, UserResponse, UserUpdate, Token, MessageResponse,
    APIKeyCreate, APIKeyResponse, APIKeyList, UsageStats, BillingInfo,
    TwoFactorSetup, TwoFactorVerify, TwoFactorDisable, PasswordChange,
    PasswordReset, PasswordResetConfirm, ErrorResponse
)
from .services import AuthService
from .security import SecurityUtils
from core.database import get_db

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()

# Dependency to get current user from JWT token
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current user from JWT token."""
    token = credentials.credentials
    payload = SecurityUtils.verify_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("user_id")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    auth_service = AuthService(db)
    user = auth_service.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

# User Registration and Authentication
@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user account."""
    auth_service = AuthService(db)
    
    try:
        user = auth_service.create_user(user_data)
        return UserResponse.from_orm(user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/login", response_model=Token)
def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user and return JWT token."""
    auth_service = AuthService(db)
    
    try:
        user = auth_service.authenticate_user(user_data.email, user_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=30)
        access_token = SecurityUtils.create_access_token(
            data={"user_id": str(user.id), "email": user.email},
            expires_delta=access_token_expires
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=30 * 60,  # 30 minutes in seconds
            user=UserResponse.from_orm(user)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/me", response_model=UserResponse)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse.from_orm(current_user)

@router.put("/me", response_model=UserResponse)
def update_current_user(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update current user information."""
    auth_service = AuthService(db)
    
    try:
        updated_user = auth_service.update_user(current_user.id, user_data)
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return UserResponse.from_orm(updated_user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

# API Key Management
@router.post("/api-keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
def create_api_key(
    api_key_data: APIKeyCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new API key for the current user."""
    auth_service = AuthService(db)
    
    try:
        api_key = auth_service.create_api_key(current_user.id, api_key_data)
        # Return the API key response (without the actual key for security)
        return APIKeyResponse.from_orm(api_key)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/api-keys", response_model=APIKeyList)
def list_api_keys(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all API keys for the current user."""
    auth_service = AuthService(db)
    api_keys = auth_service.get_user_api_keys(current_user.id)
    
    return APIKeyList(
        api_keys=[APIKeyResponse.from_orm(key) for key in api_keys],
        total=len(api_keys)
    )

@router.delete("/api-keys/{api_key_id}", response_model=MessageResponse)
def deactivate_api_key(
    api_key_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Deactivate an API key."""
    auth_service = AuthService(db)
    
    try:
        success = auth_service.deactivate_api_key(current_user.id, api_key_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )
        return MessageResponse(message="API key deactivated successfully")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

# Usage and Billing
@router.get("/usage/stats", response_model=UsageStats)
def get_usage_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get usage statistics for the current user."""
    auth_service = AuthService(db)
    return auth_service.get_user_usage_stats(current_user.id)

@router.get("/billing/info", response_model=BillingInfo)
def get_billing_info(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get billing information for the current user."""
    auth_service = AuthService(db)
    
    try:
        return auth_service.get_user_billing_info(current_user.id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

# Two-Factor Authentication
@router.post("/2fa/setup", response_model=TwoFactorSetup)
def setup_two_factor(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Setup two-factor authentication."""
    auth_service = AuthService(db)
    
    try:
        setup_data = auth_service.setup_two_factor(current_user.id)
        return TwoFactorSetup(**setup_data)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/2fa/enable", response_model=MessageResponse)
def enable_two_factor(
    setup_data: TwoFactorSetup,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enable two-factor authentication after verification."""
    auth_service = AuthService(db)
    
    try:
        auth_service.enable_two_factor(
            current_user.id,
            setup_data.secret,
            setup_data.backup_codes
        )
        return MessageResponse(message="Two-factor authentication enabled successfully")
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/2fa/verify", response_model=MessageResponse)
def verify_two_factor(
    verification_data: TwoFactorVerify,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Verify a two-factor authentication code."""
    auth_service = AuthService(db)
    
    is_valid = auth_service.verify_two_factor(current_user.id, verification_data.code)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid two-factor authentication code"
        )
    
    return MessageResponse(message="Two-factor authentication code verified successfully")

@router.post("/2fa/disable", response_model=MessageResponse)
def disable_two_factor(
    disable_data: TwoFactorDisable,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Disable two-factor authentication."""
    auth_service = AuthService(db)
    
    success = auth_service.disable_two_factor(current_user.id, disable_data.password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid password or two-factor authentication not enabled"
        )
    
    return MessageResponse(message="Two-factor authentication disabled successfully")

# Password Management
@router.post("/password/change", response_model=MessageResponse)
def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Change user password."""
    auth_service = AuthService(db)
    
    # Verify current password
    if not SecurityUtils.verify_password(password_data.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Validate new password strength
    password_validation = SecurityUtils.validate_password_strength(password_data.new_password)
    if not password_validation["is_valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password validation failed: {', '.join(password_validation['errors'])}"
        )
    
    # Update password
    hashed_password = SecurityUtils.hash_password(password_data.new_password)
    current_user.password_hash = hashed_password
    current_user.updated_at = datetime.utcnow()
    db.commit()
    
    return MessageResponse(message="Password changed successfully")

@router.post("/password/reset", response_model=MessageResponse)
def request_password_reset(password_reset_data: PasswordReset, db: Session = Depends(get_db)):
    """Request a password reset."""
    auth_service = AuthService(db)
    user = auth_service.get_user_by_email(password_reset_data.email)
    
    if user:
        # Generate reset token
        reset_token = SecurityUtils.generate_password_reset_token(password_reset_data.email)
        # In a real implementation, you would send this token via email
        # For now, we'll just return a success message
        return MessageResponse(message="Password reset email sent (if user exists)")
    
    # Always return success to prevent email enumeration
    return MessageResponse(message="Password reset email sent (if user exists)")

@router.post("/password/reset/confirm", response_model=MessageResponse)
def confirm_password_reset(password_reset_confirm_data: PasswordResetConfirm, db: Session = Depends(get_db)):
    """Confirm password reset with token."""
    # Verify reset token
    email = SecurityUtils.verify_password_reset_token(password_reset_confirm_data.token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    auth_service = AuthService(db)
    user = auth_service.get_user_by_email(email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Validate new password strength
    password_validation = SecurityUtils.validate_password_strength(password_reset_confirm_data.new_password)
    if not password_validation["is_valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password validation failed: {', '.join(password_validation['errors'])}"
        )
    
    # Update password
    hashed_password = SecurityUtils.hash_password(password_reset_confirm_data.new_password)
    user.password_hash = hashed_password
    user.updated_at = datetime.utcnow()
    db.commit()
    
    return MessageResponse(message="Password reset successfully")

# Health Check
@router.get("/health", response_model=MessageResponse)
def health_check():
    """Health check endpoint."""
    return MessageResponse(message="Authentication service is healthy") 