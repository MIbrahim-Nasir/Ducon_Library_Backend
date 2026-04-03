from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.db.models import User
from app.db.schema import UserCreate, UserLogin, Token, UserResponse, GoogleAuthToken, ConsentUpdate, ProfileUpdate
from app.auth import (
    hash_password, verify_password, create_access_token,
    get_current_user, verify_google_token,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def signup(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == payload.email))
    existing = result.scalar_one_or_none()

    if existing:
        # Account exists with Google only — no password set
        if not existing.password_hash:
            raise HTTPException(
                status_code=400,
                detail="This email is registered via Google. Please sign in with Google.",
            )
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        name=payload.name,
        email=payload.email,
        password_hash=hash_password(payload.password),
        user_consent=payload.user_consent,
        marketing_consent=payload.marketing_consent,
        phone_number=payload.phone_number,
        whatsapp_sms_consent=payload.whatsapp_sms_consent,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.post("/login", response_model=Token)
async def login(payload: UserLogin, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Account exists but was created via Google — no password
    if not user.password_hash:
        raise HTTPException(
            status_code=400,
            detail="This account uses Google sign-in. Please sign in with Google.",
        )

    if not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user_id=user.id)
    return {"access_token": token, "token_type": "bearer"}


@router.post("/google", response_model=Token)
async def google_auth(payload: GoogleAuthToken, db: AsyncSession = Depends(get_db)):
    """
    Accepts a Google ID token from the frontend (via Google Identity Services).
    - New user  → creates account (password_hash=null)
    - Returning Google user → logs in
    - Existing manual account with same email → links google_id to that account
    """
    info = verify_google_token(payload.token)

    google_id: str = info["sub"]
    email: str = info["email"]
    name: str = info.get("name", "")

    # Try find by google_id first (fast path for returning Google users)
    result = await db.execute(select(User).where(User.google_id == google_id))
    user = result.scalar_one_or_none()

    if not user:
        # Try find by email (existing manual account → link it)
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if user:
            # Link Google to existing manual account
            user.google_id = google_id
            await db.commit()
            await db.refresh(user)
        else:
            # Brand-new user via Google
            user = User(
                name=name,
                email=email,
                google_id=google_id,
                password_hash=None,
                user_consent=payload.user_consent,
                marketing_consent=payload.marketing_consent,
                phone_number=payload.phone_number,
                whatsapp_sms_consent=payload.whatsapp_sms_consent,
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)

    token = create_access_token(user_id=user.id)
    return {"access_token": token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user


@router.get("/role")
async def get_role(current_user: User = Depends(get_current_user)):
    """Returns the role of the currently authenticated user."""
    return {"role": current_user.role}


@router.get("/user_consent")
async def get_user_consent(current_user: User = Depends(get_current_user)):
    """Returns consent status for the currently authenticated user."""
    return {
        "user_consent": current_user.user_consent,
        "marketing_consent": current_user.marketing_consent,
    }


@router.patch("/user_consent", response_model=UserResponse)
async def accept_all_consent(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Accepts all consent — sets both user_consent and marketing_consent to True."""
    current_user.user_consent = True
    current_user.marketing_consent = True
    db.add(current_user)
    await db.commit()
    await db.refresh(current_user)
    return current_user


@router.put("/user_consent", response_model=UserResponse)
async def update_consent(
    payload: ConsentUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Updates user_consent and/or marketing_consent to any explicit value."""
    if payload.user_consent is not None:
        current_user.user_consent = payload.user_consent
    if payload.marketing_consent is not None:
        current_user.marketing_consent = payload.marketing_consent
    db.add(current_user)
    await db.commit()
    await db.refresh(current_user)
    return current_user


@router.put("/profile", response_model=UserResponse)
async def update_profile(
    payload: ProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Updates any combination of profile fields. Only provided fields are changed."""
    if payload.name is not None:
        current_user.name = payload.name
    if payload.phone_number is not None:
        current_user.phone_number = payload.phone_number
    if payload.whatsapp_sms_consent is not None:
        current_user.whatsapp_sms_consent = payload.whatsapp_sms_consent
    if payload.marketing_consent is not None:
        current_user.marketing_consent = payload.marketing_consent
    if payload.user_consent is not None:
        current_user.user_consent = payload.user_consent
    db.add(current_user)
    await db.commit()
    await db.refresh(current_user)
    return current_user


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_account(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Permanently deletes the currently authenticated user's account."""
    await db.delete(current_user)
    await db.commit()
