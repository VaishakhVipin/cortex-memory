# 💳 Polar.sh Integration Guide

## 🎯 **Overview**

Polar.sh is our payment gateway for Cortex Memory's pay-per-use billing system. This guide covers the complete integration setup, webhook handling, and billing flows.

---

## 🔧 **Setup Process**

### **1. Polar.sh Account Setup**

#### **Create Account**
1. Go to [polar.sh](https://polar.sh)
2. Sign up for a developer account
3. Complete organization setup
4. Navigate to API settings

#### **API Configuration**
```bash
# Get your API key from Polar.sh dashboard
POLAR_API_KEY=polar_live_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
POLAR_WEBHOOK_SECRET=whsec_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### **2. Product Configuration**

#### **Create Products in Polar.sh**
```json
{
  "products": [
    {
      "name": "Cortex Memory Free",
      "product_id": "cortex_free",
      "price_usd": 0.00,
      "tokens_per_month": 1000,
      "rate_limit": 10
    },
    {
      "name": "Cortex Memory Starter",
      "product_id": "cortex_starter", 
      "price_usd": 29.00,
      "tokens_per_month": 100000,
      "rate_limit": 100
    },
    {
      "name": "Cortex Memory Professional",
      "product_id": "cortex_professional",
      "price_usd": 99.00,
      "tokens_per_month": 1000000,
      "rate_limit": 1000
    },
    {
      "name": "Cortex Memory Enterprise",
      "product_id": "cortex_enterprise",
      "price_usd": 299.00,
      "tokens_per_month": 10000000,
      "rate_limit": 10000
    }
  ]
}
```

---

## 🔌 **API Integration**

### **1. Polar.sh Client**

#### **polar_client.py**
```python
import httpx
import hmac
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime

class PolarClient:
    def __init__(self, api_key: str, webhook_secret: str):
        self.api_key = api_key
        self.webhook_secret = webhook_secret
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
            "status": "active",
            "start_date": datetime.utcnow().isoformat()
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
            "type": "usage",
            "currency": "USD"
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
    
    async def update_subscription(self, subscription_id: str, status: str) -> Dict[str, Any]:
        """Update subscription status."""
        update_data = {"status": status}
        
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"{self.base_url}/subscriptions/{subscription_id}",
                json=update_data,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
    
    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature."""
        expected_signature = hmac.new(
            self.webhook_secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(f"sha256={expected_signature}", signature)
```

### **2. Webhook Handlers**

#### **webhooks.py**
```python
from fastapi import APIRouter, Request, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any
import json

from .polar_client import PolarClient
from ..auth.models import User, BillingEvent
from ..core.database import get_db
from ..core.config import settings

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

polar_client = PolarClient(
    api_key=settings.POLAR_API_KEY,
    webhook_secret=settings.POLAR_WEBHOOK_SECRET
)

@router.post("/polar")
async def handle_polar_webhook(request: Request, db: Session = Depends(get_db)):
    """Handle Polar.sh webhooks for subscription events."""
    
    # Get raw body for signature verification
    body = await request.body()
    
    # Verify webhook signature
    signature = request.headers.get("X-Polar-Signature")
    if not signature:
        raise HTTPException(status_code=400, detail="Missing signature")
    
    if not polar_client.verify_webhook_signature(body, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Parse event data
    event_data = json.loads(body)
    event_type = event_data.get("type")
    
    # Log billing event
    billing_event = BillingEvent(
        polar_event_id=event_data.get("id"),
        event_type=event_type,
        metadata=event_data
    )
    db.add(billing_event)
    db.commit()
    
    # Handle different event types
    if event_type == "customer.created":
        await handle_customer_created(event_data, db)
    elif event_type == "subscription.created":
        await handle_subscription_created(event_data, db)
    elif event_type == "subscription.updated":
        await handle_subscription_updated(event_data, db)
    elif event_type == "subscription.cancelled":
        await handle_subscription_cancelled(event_data, db)
    elif event_type == "invoice.paid":
        await handle_invoice_paid(event_data, db)
    elif event_type == "invoice.payment_failed":
        await handle_invoice_payment_failed(event_data, db)
    
    return {"status": "success"}

async def handle_customer_created(event_data: Dict[str, Any], db: Session):
    """Handle customer creation event."""
    customer_data = event_data.get("data", {})
    customer_id = customer_data.get("id")
    email = customer_data.get("email")
    
    # Find user by email and update Polar customer ID
    user = db.query(User).filter(User.email == email).first()
    if user:
        user.polar_customer_id = customer_id
        db.commit()

async def handle_subscription_created(event_data: Dict[str, Any], db: Session):
    """Handle subscription creation event."""
    subscription_data = event_data.get("data", {})
    customer_id = subscription_data.get("customer_id")
    product_id = subscription_data.get("product_id")
    
    # Find user by Polar customer ID
    user = db.query(User).filter(User.polar_customer_id == customer_id).first()
    if user:
        # Update subscription tier based on product
        tier_mapping = {
            "cortex_free": "free",
            "cortex_starter": "starter",
            "cortex_professional": "professional",
            "cortex_enterprise": "enterprise"
        }
        
        user.subscription_tier = tier_mapping.get(product_id, "free")
        db.commit()

async def handle_subscription_updated(event_data: Dict[str, Any], db: Session):
    """Handle subscription update event."""
    subscription_data = event_data.get("data", {})
    customer_id = subscription_data.get("customer_id")
    product_id = subscription_data.get("product_id")
    status = subscription_data.get("status")
    
    user = db.query(User).filter(User.polar_customer_id == customer_id).first()
    if user:
        if status == "active":
            tier_mapping = {
                "cortex_free": "free",
                "cortex_starter": "starter",
                "cortex_professional": "professional",
                "cortex_enterprise": "enterprise"
            }
            user.subscription_tier = tier_mapping.get(product_id, "free")
        elif status == "cancelled":
            user.subscription_tier = "free"
        
        db.commit()

async def handle_subscription_cancelled(event_data: Dict[str, Any], db: Session):
    """Handle subscription cancellation event."""
    subscription_data = event_data.get("data", {})
    customer_id = subscription_data.get("customer_id")
    
    user = db.query(User).filter(User.polar_customer_id == customer_id).first()
    if user:
        user.subscription_tier = "free"
        db.commit()

async def handle_invoice_paid(event_data: Dict[str, Any], db: Session):
    """Handle invoice payment event."""
    invoice_data = event_data.get("data", {})
    customer_id = invoice_data.get("customer_id")
    amount_usd = invoice_data.get("amount_usd", 0)
    
    # Log successful payment
    billing_event = BillingEvent(
        polar_event_id=event_data.get("id"),
        event_type="invoice.paid",
        amount_usd=amount_usd,
        status="completed",
        metadata=event_data
    )
    db.add(billing_event)
    db.commit()

async def handle_invoice_payment_failed(event_data: Dict[str, Any], db: Session):
    """Handle failed payment event."""
    invoice_data = event_data.get("data", {})
    customer_id = invoice_data.get("customer_id")
    
    # Find user and potentially downgrade
    user = db.query(User).filter(User.polar_customer_id == customer_id).first()
    if user and user.subscription_tier != "free":
        user.subscription_tier = "free"
        db.commit()
```

---

## 💰 **Billing Service**

### **1. Usage-Based Billing**

#### **billing_service.py**
```python
from sqlalchemy.orm import Session
from typing import Dict, Any
from datetime import datetime, timedelta

from .polar_client import PolarClient
from ..auth.models import User, UsageLog, BillingEvent
from ..core.config import settings

class BillingService:
    def __init__(self, db: Session):
        self.db = db
        self.polar_client = PolarClient(
            api_key=settings.POLAR_API_KEY,
            webhook_secret=settings.POLAR_WEBHOOK_SECRET
        )
    
    async def calculate_usage_cost(self, user_id: str, tokens_used: int) -> float:
        """Calculate cost for token usage."""
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            return 0.0
        
        # Cost per token based on tier
        cost_per_token = {
            "free": 0.0002,      # $0.0002 per token
            "starter": 0.0001,   # $0.0001 per token
            "professional": 0.00005,  # $0.00005 per token
            "enterprise": 0.00002     # $0.00002 per token
        }
        
        rate = cost_per_token.get(user.subscription_tier, 0.0002)
        return tokens_used * rate
    
    async def process_usage_billing(self, user_id: str, tokens_used: int, operation: str):
        """Process usage and create billing charges."""
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user or not user.polar_customer_id:
            return
        
        # Calculate cost
        cost_usd = await self.calculate_usage_cost(user_id, tokens_used)
        
        # Check if we should bill (accumulate small charges)
        monthly_usage = await self.get_monthly_usage(user_id)
        monthly_cost = await self.calculate_usage_cost(user_id, monthly_usage)
        
        # Bill if cost exceeds threshold or monthly billing cycle
        if monthly_cost >= 1.00 or await self.is_monthly_billing_cycle(user_id):
            await self.create_billing_charge(user, monthly_cost, f"Monthly usage: {monthly_usage} tokens")
            
            # Reset monthly usage
            await self.reset_monthly_usage(user_id)
    
    async def create_billing_charge(self, user: User, amount_usd: float, description: str):
        """Create billing charge in Polar.sh."""
        try:
            charge = await self.polar_client.create_usage_charge(
                customer_id=user.polar_customer_id,
                amount_usd=amount_usd,
                description=description
            )
            
            # Log billing event
            billing_event = BillingEvent(
                polar_event_id=charge.get("id"),
                event_type="usage_charge",
                amount_usd=amount_usd,
                status="pending",
                metadata=charge
            )
            self.db.add(billing_event)
            self.db.commit()
            
            return charge
        except Exception as e:
            # Log billing failure
            billing_event = BillingEvent(
                event_type="usage_charge_failed",
                amount_usd=amount_usd,
                status="failed",
                metadata={"error": str(e)}
            )
            self.db.add(billing_event)
            self.db.commit()
            raise
    
    async def get_monthly_usage(self, user_id: str) -> int:
        """Get total usage for current month."""
        start_of_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        usage = self.db.query(UsageLog).filter(
            UsageLog.user_id == user_id,
            UsageLog.created_at >= start_of_month
        ).all()
        
        return sum(log.tokens_used for log in usage)
    
    async def is_monthly_billing_cycle(self, user_id: str) -> bool:
        """Check if it's time for monthly billing."""
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            return False
        
        # Check if it's the first of the month or billing day
        now = datetime.utcnow()
        return now.day == 1 or now.day == user.billing_day
    
    async def reset_monthly_usage(self, user_id: str):
        """Reset monthly usage counters."""
        # This could be implemented by marking usage logs as billed
        # or by maintaining separate billing period records
        pass
```

### **2. Subscription Management**

#### **subscription_service.py**
```python
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
from datetime import datetime

from .polar_client import PolarClient
from ..auth.models import User
from ..core.config import settings

class SubscriptionService:
    def __init__(self, db: Session):
        self.db = db
        self.polar_client = PolarClient(
            api_key=settings.POLAR_API_KEY,
            webhook_secret=settings.POLAR_WEBHOOK_SECRET
        )
    
    async def create_customer(self, user: User) -> str:
        """Create customer in Polar.sh."""
        if user.polar_customer_id:
            return user.polar_customer_id
        
        customer_id = await self.polar_client.create_customer(
            email=user.email,
            name=f"{user.first_name} {user.last_name}",
            metadata={
                "user_id": str(user.id),
                "company": user.company
            }
        )
        
        user.polar_customer_id = customer_id
        self.db.commit()
        
        return customer_id
    
    async def create_subscription(self, user: User, product_id: str) -> str:
        """Create subscription in Polar.sh."""
        # Ensure customer exists
        customer_id = await self.create_customer(user)
        
        # Create subscription
        subscription_id = await self.polar_client.create_subscription(
            customer_id=customer_id,
            product_id=product_id
        )
        
        return subscription_id
    
    async def upgrade_subscription(self, user: User, new_product_id: str):
        """Upgrade user subscription."""
        if not user.polar_customer_id:
            await self.create_customer(user)
        
        # Create new subscription (Polar.sh handles the upgrade)
        subscription_id = await self.create_subscription(user, new_product_id)
        
        return subscription_id
    
    async def cancel_subscription(self, user: User):
        """Cancel user subscription."""
        if not user.polar_customer_id:
            return
        
        # Get active subscription
        customer = await self.polar_client.get_customer(user.polar_customer_id)
        subscriptions = customer.get("subscriptions", [])
        
        for subscription in subscriptions:
            if subscription.get("status") == "active":
                await self.polar_client.update_subscription(
                    subscription_id=subscription["id"],
                    status="cancelled"
                )
                break
```

---

## 📊 **Usage Analytics**

### **1. Billing Reports**

#### **analytics_service.py**
```python
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, Any, List
from datetime import datetime, timedelta

from ..auth.models import User, UsageLog, BillingEvent

class AnalyticsService:
    def __init__(self, db: Session):
        self.db = db
    
    async def get_user_usage_stats(self, user_id: str, period: str = "month") -> Dict[str, Any]:
        """Get usage statistics for user."""
        if period == "month":
            start_date = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif period == "week":
            start_date = datetime.utcnow() - timedelta(days=7)
        else:
            start_date = datetime.utcnow() - timedelta(days=30)
        
        usage_logs = self.db.query(UsageLog).filter(
            UsageLog.user_id == user_id,
            UsageLog.created_at >= start_date
        ).all()
        
        total_tokens = sum(log.tokens_used for log in usage_logs)
        total_cost = sum(float(log.cost_usd) for log in usage_logs)
        total_requests = len(usage_logs)
        
        return {
            "period": period,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "total_requests": total_requests,
            "average_tokens_per_request": total_tokens / total_requests if total_requests > 0 else 0
        }
    
    async def get_billing_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get billing history for user."""
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user or not user.polar_customer_id:
            return []
        
        billing_events = self.db.query(BillingEvent).filter(
            BillingEvent.metadata.contains({"customer_id": user.polar_customer_id})
        ).order_by(BillingEvent.created_at.desc()).all()
        
        return [
            {
                "id": event.id,
                "event_type": event.event_type,
                "amount_usd": float(event.amount_usd) if event.amount_usd else 0,
                "status": event.status,
                "created_at": event.created_at.isoformat(),
                "description": event.metadata.get("description", "")
            }
            for event in billing_events
        ]
    
    async def get_revenue_analytics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get revenue analytics for admin dashboard."""
        billing_events = self.db.query(BillingEvent).filter(
            BillingEvent.created_at >= start_date,
            BillingEvent.created_at <= end_date,
            BillingEvent.status == "completed"
        ).all()
        
        total_revenue = sum(float(event.amount_usd) for event in billing_events if event.amount_usd)
        
        # Revenue by tier
        revenue_by_tier = {}
        for event in billing_events:
            if event.metadata and "product_id" in event.metadata:
                product_id = event.metadata["product_id"]
                amount = float(event.amount_usd) if event.amount_usd else 0
                revenue_by_tier[product_id] = revenue_by_tier.get(product_id, 0) + amount
        
        return {
            "total_revenue": total_revenue,
            "revenue_by_tier": revenue_by_tier,
            "total_transactions": len(billing_events),
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            }
        }
```

---

## 🔧 **Configuration**

### **1. Environment Variables**
```bash
# .env
POLAR_API_KEY=polar_live_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
POLAR_WEBHOOK_SECRET=whsec_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
POLAR_WEBHOOK_URL=https://yourdomain.com/api/v1/webhooks/polar

# Billing configuration
BILLING_THRESHOLD_USD=1.00
MONTHLY_BILLING_DAY=1
COST_PER_TOKEN_FREE=0.0002
COST_PER_TOKEN_STARTER=0.0001
COST_PER_TOKEN_PROFESSIONAL=0.00005
COST_PER_TOKEN_ENTERPRISE=0.00002
```

### **2. Database Schema**
```sql
-- Add billing events table
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

-- Add billing day to users table
ALTER TABLE users ADD COLUMN billing_day INTEGER DEFAULT 1;
```

---

## 🧪 **Testing**

### **1. Webhook Testing**
```python
# tests/test_polar_webhooks.py
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

def test_polar_webhook_signature_verification(client: TestClient):
    """Test webhook signature verification."""
    
    # Create test payload
    payload = {
        "type": "subscription.created",
        "data": {
            "id": "sub_123",
            "customer_id": "cus_456",
            "product_id": "cortex_starter"
        }
    }
    
    # Mock signature verification
    with patch('billing.polar_client.PolarClient.verify_webhook_signature', return_value=True):
        response = client.post(
            "/api/v1/webhooks/polar",
            json=payload,
            headers={"X-Polar-Signature": "test_signature"}
        )
        
        assert response.status_code == 200
        assert response.json()["status"] == "success"

def test_billing_charge_creation(client: TestClient):
    """Test usage-based billing charge creation."""
    
    # Mock Polar.sh API calls
    with patch('billing.polar_client.PolarClient.create_usage_charge') as mock_charge:
        mock_charge.return_value = {
            "id": "ch_123",
            "amount_usd": 1.50,
            "status": "pending"
        }
        
        # Test usage tracking
        response = client.post(
            "/api/v1/context/generate",
            json={"prompt": "Test prompt"},
            headers={"X-API-Key": "test_api_key"}
        )
        
        assert response.status_code == 200
        # Verify billing charge was created
        mock_charge.assert_called_once()
```

---

## 🚀 **Deployment Checklist**

### **Pre-Deployment**
- [ ] Polar.sh account created and configured
- [ ] Products created in Polar.sh dashboard
- [ ] Webhook endpoint configured in Polar.sh
- [ ] API keys and webhook secrets configured
- [ ] Database schema updated with billing tables
- [ ] Environment variables set

### **Post-Deployment**
- [ ] Webhook delivery verified
- [ ] Test subscription created
- [ ] Usage tracking tested
- [ ] Billing charges verified
- [ ] Payment processing tested
- [ ] Analytics dashboard working

---

## 📚 **Resources**

- [Polar.sh API Documentation](https://docs.polar.sh/)
- [Webhook Security Best Practices](https://polar.sh/docs/webhooks)
- [Billing Integration Guide](https://polar.sh/docs/billing)
- [Testing Webhooks](https://polar.sh/docs/webhooks#testing)

---

**🎯 This guide provides everything needed to integrate Polar.sh billing with Cortex Memory!** 