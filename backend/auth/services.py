"""
🧠 Authentication Services
Business logic for user authentication, API key management, and usage tracking
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from .models import User, APIKey, UsageLog, BillingEvent
from .schemas import UserCreate, UserUpdate, APIKeyCreate, UsageStats, BillingInfo
from .security import SecurityUtils

class AuthService:
    """Authentication service for user management and API operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    # User Management
    def create_user(self, user_data: UserCreate) -> User:
        """Create a new user account."""
        # Check if user already exists
        existing_user = self.db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise ValueError("User with this email already exists")
        
        # Validate password strength
        password_validation = SecurityUtils.validate_password_strength(user_data.password)
        if not password_validation["is_valid"]:
            raise ValueError(f"Password validation failed: {', '.join(password_validation['errors'])}")
        
        # Create user
        hashed_password = SecurityUtils.hash_password(user_data.password)
        user = User(
            email=SecurityUtils.sanitize_email(user_data.email),
            password_hash=hashed_password,
            subscription_tier="free"
        )
        
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate a user with email and password."""
        user = self.db.query(User).filter(User.email == SecurityUtils.sanitize_email(email)).first()
        if not user:
            return None
        
        if not SecurityUtils.verify_password(password, user.password_hash):
            return None
        
        if not user.is_active:
            raise ValueError("Account is deactivated")
        
        return user
    
    def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        return self.db.query(User).filter(User.id == user_id).first()
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.db.query(User).filter(User.email == SecurityUtils.sanitize_email(email)).first()
    
    def update_user(self, user_id: UUID, user_data: UserUpdate) -> Optional[User]:
        """Update user information."""
        user = self.get_user_by_id(user_id)
        if not user:
            return None
        
        # Update fields
        if user_data.email is not None:
            # Check if email is already taken
            existing_user = self.db.query(User).filter(
                and_(User.email == SecurityUtils.sanitize_email(user_data.email), User.id != user_id)
            ).first()
            if existing_user:
                raise ValueError("Email is already taken")
            user.email = SecurityUtils.sanitize_email(user_data.email)
        
        if user_data.subscription_tier is not None:
            user.subscription_tier = user_data.subscription_tier
        
        if user_data.is_active is not None:
            user.is_active = user_data.is_active
        
        user.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(user)
        return user
    
    def delete_user(self, user_id: UUID) -> bool:
        """Delete a user account."""
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        self.db.delete(user)
        self.db.commit()
        return True
    
    # API Key Management
    def create_api_key(self, user_id: UUID, api_key_data: APIKeyCreate) -> APIKey:
        """Create a new API key for a user."""
        user = self.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        
        # Generate API key
        api_key_value = SecurityUtils.generate_api_key()
        api_key_hash = SecurityUtils.hash_api_key(api_key_value)
        
        # Create API key record
        api_key = APIKey(
            user_id=user_id,
            key_hash=api_key_hash,
            name=api_key_data.name or "Default API Key"
        )
        
        self.db.add(api_key)
        self.db.commit()
        self.db.refresh(api_key)
        
        # Return the actual API key value (only shown once)
        api_key.actual_key = api_key_value
        return api_key
    
    def get_api_key_by_hash(self, key_hash: str) -> Optional[APIKey]:
        """Get API key by its hash."""
        return self.db.query(APIKey).filter(
            and_(APIKey.key_hash == key_hash, APIKey.is_active == True)
        ).first()
    
    def get_user_api_keys(self, user_id: UUID) -> List[APIKey]:
        """Get all API keys for a user."""
        return self.db.query(APIKey).filter(APIKey.user_id == user_id).all()
    
    def deactivate_api_key(self, user_id: UUID, api_key_id: UUID) -> bool:
        """Deactivate an API key."""
        api_key = self.db.query(APIKey).filter(
            and_(APIKey.id == api_key_id, APIKey.user_id == user_id)
        ).first()
        
        if not api_key:
            return False
        
        api_key.is_active = False
        self.db.commit()
        return True
    
    def update_api_key_last_used(self, api_key_id: UUID):
        """Update the last used timestamp for an API key."""
        api_key = self.db.query(APIKey).filter(APIKey.id == api_key_id).first()
        if api_key:
            api_key.last_used_at = datetime.utcnow()
            self.db.commit()
    
    # Usage Tracking
    def log_api_usage(self, user_id: UUID, api_key_id: Optional[UUID], 
                     endpoint: str, tokens_used: int, cost_usd: float,
                     provider: Optional[str] = None, response_time_ms: Optional[int] = None,
                     status_code: Optional[int] = None, error_message: Optional[str] = None) -> UsageLog:
        """Log API usage for billing and analytics."""
        usage_log = UsageLog(
            user_id=user_id,
            api_key_id=api_key_id,
            endpoint=endpoint,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            provider=provider,
            response_time_ms=response_time_ms,
            status_code=status_code,
            error_message=error_message
        )
        
        self.db.add(usage_log)
        self.db.commit()
        self.db.refresh(usage_log)
        return usage_log
    
    def get_user_usage_stats(self, user_id: UUID) -> UsageStats:
        """Get usage statistics for a user."""
        # Total usage
        total_stats = self.db.query(
            func.count(UsageLog.id).label('total_requests'),
            func.sum(UsageLog.tokens_used).label('total_tokens'),
            func.sum(UsageLog.cost_usd).label('total_cost')
        ).filter(UsageLog.user_id == user_id).first()
        
        # Today's usage
        today = datetime.utcnow().date()
        today_stats = self.db.query(
            func.count(UsageLog.id).label('requests_today'),
            func.sum(UsageLog.tokens_used).label('tokens_today'),
            func.sum(UsageLog.cost_usd).label('cost_today')
        ).filter(
            and_(
                UsageLog.user_id == user_id,
                func.date(UsageLog.created_at) == today
            )
        ).first()
        
        # This month's usage
        month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_stats = self.db.query(
            func.count(UsageLog.id).label('requests_month'),
            func.sum(UsageLog.tokens_used).label('tokens_month'),
            func.sum(UsageLog.cost_usd).label('cost_month')
        ).filter(
            and_(
                UsageLog.user_id == user_id,
                UsageLog.created_at >= month_start
            )
        ).first()
        
        return UsageStats(
            total_requests=total_stats.total_requests or 0,
            total_tokens=total_stats.total_tokens or 0,
            total_cost_usd=float(total_stats.total_cost or 0),
            requests_today=today_stats.requests_today or 0,
            tokens_today=today_stats.tokens_today or 0,
            cost_today_usd=float(today_stats.cost_today or 0),
            requests_this_month=month_stats.requests_month or 0,
            tokens_this_month=month_stats.tokens_month or 0,
            cost_this_month_usd=float(month_stats.cost_month or 0)
        )
    
    def get_user_usage_logs(self, user_id: UUID, limit: int = 100, offset: int = 0) -> List[UsageLog]:
        """Get usage logs for a user."""
        return self.db.query(UsageLog).filter(UsageLog.user_id == user_id)\
            .order_by(UsageLog.created_at.desc())\
            .offset(offset).limit(limit).all()
    
    # Billing and Subscription
    def get_user_billing_info(self, user_id: UUID) -> BillingInfo:
        """Get billing information for a user."""
        user = self.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        
        # Get monthly usage
        month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        monthly_usage = self.db.query(func.sum(UsageLog.tokens_used)).filter(
            and_(
                UsageLog.user_id == user_id,
                UsageLog.created_at >= month_start
            )
        ).scalar() or 0
        
        # Define tier limits
        tier_limits = {
            "free": 1000,
            "starter": 10000,
            "professional": 100000,
            "enterprise": 1000000
        }
        
        monthly_limit = tier_limits.get(user.subscription_tier, 1000)
        remaining_usage = max(0, monthly_limit - monthly_usage)
        
        # Calculate total cost this month
        monthly_cost = self.db.query(func.sum(UsageLog.cost_usd)).filter(
            and_(
                UsageLog.user_id == user_id,
                UsageLog.created_at >= month_start
            )
        ).scalar() or 0.0
        
        return BillingInfo(
            subscription_tier=user.subscription_tier,
            monthly_limit=monthly_limit,
            current_usage=monthly_usage,
            remaining_usage=remaining_usage,
            next_billing_date=month_start + timedelta(days=32),  # Approximate
            total_cost_this_month=float(monthly_cost)
        )
    
    def check_usage_limit(self, user_id: UUID) -> bool:
        """Check if user has exceeded their usage limit."""
        billing_info = self.get_user_billing_info(user_id)
        return billing_info.remaining_usage > 0
    
    # Two-Factor Authentication
    def setup_two_factor(self, user_id: UUID) -> Dict[str, Any]:
        """Setup two-factor authentication for a user."""
        user = self.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        
        # Generate 2FA secret
        secret = SecurityUtils.generate_two_factor_secret()
        qr_code_url = SecurityUtils.generate_two_factor_qr_code(secret, user.email)
        backup_codes = SecurityUtils.generate_backup_codes()
        
        # Store secret temporarily (user needs to verify before saving)
        return {
            "secret": secret,
            "qr_code_url": qr_code_url,
            "backup_codes": backup_codes
        }
    
    def enable_two_factor(self, user_id: UUID, secret: str, backup_codes: List[str]):
        """Enable two-factor authentication for a user."""
        user = self.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        
        user.two_factor_secret = secret
        user.two_factor_enabled = True
        # In a real implementation, you'd store backup codes securely
        self.db.commit()
    
    def verify_two_factor(self, user_id: UUID, code: str) -> bool:
        """Verify a two-factor authentication code."""
        user = self.get_user_by_id(user_id)
        if not user or not user.two_factor_enabled or not user.two_factor_secret:
            return False
        
        return SecurityUtils.verify_two_factor_code(user.two_factor_secret, code)
    
    def disable_two_factor(self, user_id: UUID, password: str) -> bool:
        """Disable two-factor authentication for a user."""
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        if not SecurityUtils.verify_password(password, user.password_hash):
            return False
        
        user.two_factor_enabled = False
        user.two_factor_secret = None
        self.db.commit()
        return True 