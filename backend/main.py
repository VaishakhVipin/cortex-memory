"""
🧠 Cortex Memory Backend
Main FastAPI application with authentication and API key management
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from auth.routes import router as auth_router
from auth.services import AuthService
from auth.security import SecurityUtils
from core.database import get_db, create_tables, check_database_connection
from cortex_memory import CortexClient

load_dotenv()

# Application metadata
APP_NAME = "Cortex Memory Backend"
APP_VERSION = "2.0.3"
APP_DESCRIPTION = "🧠 The Smart Context Layer for Prompt Chains in LLMs - Enterprise-grade context-aware AI system"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print("🚀 Starting Cortex Memory Backend...")
    
    # Check database connection
    if not check_database_connection():
        print("❌ Database connection failed!")
        raise Exception("Database connection failed")
    
    print("✅ Database connection established")
    
    # Create tables if they don't exist
    try:
        create_tables()
        print("✅ Database tables created/verified")
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")
        raise e
    
    print(f"🎉 {APP_NAME} v{APP_VERSION} started successfully!")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Cortex Memory Backend...")

# Create FastAPI application
app = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key middleware for protected endpoints
security = HTTPBearer(auto_error=False)

async def get_api_key_user(request: Request, db: Session = Depends(get_db)):
    """Get user from API key for protected endpoints."""
    # Skip authentication for certain endpoints
    if request.url.path in ["/docs", "/redoc", "/openapi.json", "/auth/health"]:
        return None
    
    # Check for API key in headers
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify API key
    auth_service = AuthService(db)
    api_key_hash = SecurityUtils.hash_api_key(api_key)
    api_key_record = auth_service.get_api_key_by_hash(api_key_hash)
    
    if not api_key_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user
    user = auth_service.get_user_by_id(api_key_record.user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check usage limits
    if not auth_service.check_usage_limit(user.id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Usage limit exceeded for your subscription tier"
        )
    
    # Update last used timestamp
    auth_service.update_api_key_last_used(api_key_record.id)
    
    return user

# Include authentication routes
app.include_router(auth_router, prefix="/api/v1")

# Cortex Memory API endpoints
@app.post("/api/v1/generate")
async def generate_with_context(
    request: Request,
    user_id: str,
    prompt: str,
    provider: str = "auto",
    db: Session = Depends(get_db),
    current_user = Depends(get_api_key_user)
):
    """Generate context-aware response using Cortex Memory."""
    try:
        # Initialize Cortex client
        client = CortexClient(api_key="internal")  # Internal API key for backend
        
        # Generate response
        response = client.generate_with_context(
            user_id=user_id,
            prompt=prompt,
            provider=provider
        )
        
        # Log usage
        auth_service = AuthService(db)
        auth_service.log_api_usage(
            user_id=current_user.id,
            api_key_id=None,  # Will be set by middleware
            endpoint="/api/v1/generate",
            tokens_used=len(prompt.split()) + len(response.split()),  # Approximate
            cost_usd=0.001,  # Approximate cost
            provider=provider,
            status_code=200
        )
        
        return {"response": response, "provider": provider}
        
    except Exception as e:
        # Log error
        auth_service = AuthService(db)
        auth_service.log_api_usage(
            user_id=current_user.id,
            api_key_id=None,
            endpoint="/api/v1/generate",
            tokens_used=0,
            cost_usd=0.0,
            provider=provider,
            status_code=500,
            error_message=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": APP_NAME,
        "version": APP_VERSION,
        "database": "connected" if check_database_connection() else "disconnected"
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to Cortex Memory Backend",
        "version": APP_VERSION,
        "description": APP_DESCRIPTION,
        "docs": "/docs",
        "health": "/api/v1/health"
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": request.url.path
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG", "False").lower() == "true" else "An unexpected error occurred",
            "status_code": 500,
            "path": request.url.path
        }
    )

if __name__ == "__main__":
    import uvicorn
    import socket
    
    def find_free_port(start_port=8000, max_attempts=10):
        """Find a free port starting from start_port."""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        return start_port  # Fallback to original port
    
    # Find a free port
    port = find_free_port()
    print(f"🚀 Starting server on port {port}")
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",  # Changed from 0.0.0.0 to 127.0.0.1 for security
        port=port,
        reload=True,
        log_level="info"
    ) 