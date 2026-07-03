from fastapi import APIRouter, Depends, HTTPException, Request, status, Header
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.db.models import User
from app.db.schema import (
    UserCreate,
    UserLogin,
    Token,
    UserResponse,
    GoogleAuthToken,
    ConsentUpdate,
    ProfileUpdate,
    OtpMessageResponse,
    SignupVerifyRequest,
    PasswordForgotRequest,
    PasswordVerifyOtpRequest,
    PasswordResetRequest,
    PasswordResetTokenResponse,
)
from app.auth import (
    hash_password,
    verify_password,
    create_access_token,
    create_password_reset_token,
    get_current_user,
    verify_google_token,
    revoke_token,
    decode_token_payload,
    get_optional_user,
)
from app.rate_limiter import require_rate_limit
from app.otp_service import (
    issue_otp,
    verify_otp,
    normalize_email,
    PURPOSE_SIGNUP,
    PURPOSE_PASSWORD_RESET,
    OTP_RESEND_COOLDOWN_SECONDS,
)

router = APIRouter(prefix="/auth", tags=["auth"])
_oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

_GENERIC_OTP_MESSAGE = (
    "If an account exists with that email, a verification code has been sent."
)


@router.post("/signup", status_code=status.HTTP_410_GONE)
async def signup_legacy():
    """Direct signup is disabled — use /auth/signup/request then /auth/signup/verify."""
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="Signup requires email verification. Use /auth/signup/request and /auth/signup/verify.",
    )


@router.post("/signup/request", response_model=OtpMessageResponse)
async def signup_request(
    request: Request,
    payload: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    require_rate_limit(request, max_requests=5, window_seconds=300, key_prefix="signup_otp")
    email = normalize_email(payload.email)

    result = await db.execute(select(User).where(User.email == email))
    existing = result.scalar_one_or_none()
    if existing:
        if not existing.password_hash:
            raise HTTPException(
                status_code=400,
                detail="This email is registered via Google. Please sign in with Google.",
            )
        raise HTTPException(status_code=400, detail="Email already registered")

    pending_data = {
        "name": payload.name,
        "password_hash": hash_password(payload.password),
        "user_consent": payload.user_consent,
        "marketing_consent": payload.marketing_consent,
        "phone_number": payload.phone_number,
        "whatsapp_sms_consent": payload.whatsapp_sms_consent,
    }

    try:
        await issue_otp(
            db,
            email=email,
            purpose=PURPOSE_SIGNUP,
            pending_data=pending_data,
        )
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to send verification email. Please try again later.",
        )

    return OtpMessageResponse(
        message="Verification code sent. Check your email to complete signup.",
        cooldown_seconds=OTP_RESEND_COOLDOWN_SECONDS,
    )


@router.post("/signup/verify", response_model=Token)
async def signup_verify(
    request: Request,
    payload: SignupVerifyRequest,
    db: AsyncSession = Depends(get_db),
):
    require_rate_limit(request, max_requests=10, window_seconds=300, key_prefix="signup_verify")
    email = normalize_email(payload.email)

    result = await db.execute(select(User).where(User.email == email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    record = await verify_otp(
        db,
        email=email,
        otp=payload.otp,
        purpose=PURPOSE_SIGNUP,
    )

    pending = record.pending_data or {}
    if not pending.get("name") or not pending.get("password_hash"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification code.",
        )

    user = User(
        name=pending["name"],
        email=email,
        password_hash=pending["password_hash"],
        email_verified=True,
        user_consent=bool(pending.get("user_consent")),
        marketing_consent=bool(pending.get("marketing_consent")),
        phone_number=pending.get("phone_number"),
        whatsapp_sms_consent=bool(pending.get("whatsapp_sms_consent")),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    token = create_access_token(user_id=user.id)
    return {"access_token": token, "token_type": "bearer"}


@router.post("/password/forgot", response_model=OtpMessageResponse)
async def password_forgot(
    request: Request,
    payload: PasswordForgotRequest,
    db: AsyncSession = Depends(get_db),
):
    require_rate_limit(request, max_requests=5, window_seconds=300, key_prefix="password_forgot")
    email = normalize_email(payload.email)

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user and user.password_hash:
        try:
            await issue_otp(db, email=email, purpose=PURPOSE_PASSWORD_RESET)
        except HTTPException:
            raise
        except RuntimeError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Unable to send verification email. Please try again later.",
            )

    return OtpMessageResponse(
        message=_GENERIC_OTP_MESSAGE,
        cooldown_seconds=OTP_RESEND_COOLDOWN_SECONDS,
    )


@router.post("/password/verify-otp", response_model=PasswordResetTokenResponse)
async def password_verify_otp(
    request: Request,
    payload: PasswordVerifyOtpRequest,
    db: AsyncSession = Depends(get_db),
):
    require_rate_limit(request, max_requests=10, window_seconds=300, key_prefix="password_verify_otp")
    email = normalize_email(payload.email)

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user or not user.password_hash:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification code.",
        )

    await verify_otp(
        db,
        email=email,
        otp=payload.otp,
        purpose=PURPOSE_PASSWORD_RESET,
    )

    reset_token = create_password_reset_token(email)
    return PasswordResetTokenResponse(reset_token=reset_token)


@router.post("/password/reset", response_model=OtpMessageResponse)
async def password_reset(
    request: Request,
    payload: PasswordResetRequest,
    db: AsyncSession = Depends(get_db),
):
    require_rate_limit(request, max_requests=5, window_seconds=300, key_prefix="password_reset")
    email = normalize_email(payload.email)

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user or not user.password_hash:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification code.",
        )

    await verify_otp(
        db,
        email=email,
        otp=payload.otp,
        purpose=PURPOSE_PASSWORD_RESET,
    )

    user.password_hash = hash_password(payload.new_password)
    db.add(user)
    await db.commit()

    return OtpMessageResponse(message="Password updated successfully. You can now log in.")


@router.post("/login", response_model=Token)
async def login(request: Request, payload: UserLogin, db: AsyncSession = Depends(get_db)):
    require_rate_limit(request, max_requests=8, window_seconds=60, key_prefix="login")
    email = normalize_email(payload.email)
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Account exists but was created via Google — no password
    if not user.password_hash:
        raise HTTPException(
            status_code=400,
            detail="This account uses Google sign-in. Please sign in with Google.",
        )

    if not user.email_verified:
        raise HTTPException(
            status_code=403,
            detail="Email not verified. Please complete signup verification.",
        )

    if not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user_id=user.id)
    return {"access_token": token, "token_type": "bearer"}


@router.post("/google", response_model=Token)
async def google_auth(request: Request, payload: GoogleAuthToken, db: AsyncSession = Depends(get_db)):
    require_rate_limit(request, max_requests=10, window_seconds=60, key_prefix="google_auth")
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
            user.email_verified = True
            await db.commit()
            await db.refresh(user)
        else:
            # Brand-new user via Google
            user = User(
                name=name,
                email=email,
                google_id=google_id,
                password_hash=None,
                email_verified=True,
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


@router.post("/heartbeat", status_code=204)
async def session_heartbeat(
    current_user: User | None = Depends(get_optional_user),
    x_guest_session_id: str | None = Header(None, alias="X-Guest-Session-Id"),
):
    """Lightweight activity ping for time-on-platform analytics (non-blocking)."""
    from app.admin.usage_recorder import heartbeat as record_heartbeat
    record_heartbeat(
        user_id=current_user.id if current_user else None,
        guest_session_id=x_guest_session_id,
    )


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


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(token: str = Depends(_oauth2)):
    """Revoke the current Bearer token so it can no longer be used."""
    if token:
        payload = decode_token_payload(token)
        jti = payload.get("jti")
        exp = payload.get("exp", 0)
        if jti:
            revoke_token(jti, float(exp))


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_account(
    token: str = Depends(_oauth2),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Permanently deletes the currently authenticated user's account and revokes the token."""
    if token:
        payload = decode_token_payload(token)
        jti = payload.get("jti")
        exp = payload.get("exp", 0)
        if jti:
            revoke_token(jti, float(exp))
    await db.delete(current_user)
    await db.commit()
