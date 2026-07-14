"""Pytest fixtures for Selenium E2E against Ducon production."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService

from tests.e2e.config import FALLBACK_BASE_URL, E2EConfig, load_config
from tests.e2e.pages.auth import AuthPage
from tests.e2e.pages.chat import ChatPage
from tests.e2e.pages.home import HomePage
from tests.e2e.pages.sidebar import SidebarPage
from tests.e2e.pages.studio import StudioPage

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_PNG = FIXTURES_DIR / "sample_upload.png"


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: Selenium end-to-end tests (excluded from default CI)")
    config.addinivalue_line("markers", "e2e_gen: E2E tests that may trigger paid AI generation")


def pytest_collection_modifyitems(config, items):
    """Keep `pytest tests/` free of e2e unless explicitly selected with -m e2e."""
    markexpr = (config.option.markexpr or "").strip()
    if "e2e" in markexpr:
        return
    # If user ran the e2e path directly, allow without -m
    args = " ".join(str(a) for a in config.args).replace("\\", "/")
    if "tests/e2e" in args or "tests\\e2e" in args:
        return
    skip = pytest.mark.skip(reason="E2E tests require: pytest -m e2e tests/e2e")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def e2e_config() -> E2EConfig:
    return load_config()


@pytest.fixture(scope="session")
def sample_upload_png() -> Path:
    assert SAMPLE_PNG.exists(), f"Missing fixture image: {SAMPLE_PNG}"
    return SAMPLE_PNG


def _build_chrome(config: E2EConfig) -> webdriver.Chrome:
    options = ChromeOptions()
    if config.headless:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1440,900")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-notifications")
    # Reduce automation fingerprints slightly (Turnstile may still block)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    # Prefer Selenium Manager (bundled with Selenium 4.6+); fall back to webdriver-manager.
    try:
        driver = webdriver.Chrome(options=options)
    except Exception:
        from webdriver_manager.chrome import ChromeDriverManager

        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

    driver.set_page_load_timeout(config.page_load_timeout)
    if config.implicit_wait:
        driver.implicitly_wait(config.implicit_wait)
    return driver


def _probe_base_url(driver: webdriver.Chrome, config: E2EConfig) -> str:
    """Use configured URL; if DNS/connect fails, try app.ducon.com fallback once."""
    primary = config.base_url
    try:
        driver.get(primary + "/")
        return primary
    except Exception as primary_err:
        if primary.rstrip("/") == FALLBACK_BASE_URL.rstrip("/"):
            raise
        fallback = FALLBACK_BASE_URL
        try:
            driver.get(fallback + "/")
            os.environ["E2E_RESOLVED_BASE_URL"] = fallback
            print(
                f"[e2e] Primary base URL failed ({primary}: {primary_err}). "
                f"Using fallback: {fallback}"
            )
            return fallback
        except Exception:
            raise primary_err


@pytest.fixture
def driver(e2e_config: E2EConfig):
    drv = _build_chrome(e2e_config)
    resolved = _probe_base_url(drv, e2e_config)
    drv.e2e_base_url = resolved  # type: ignore[attr-defined]
    yield drv
    drv.quit()


@pytest.fixture
def config(driver, e2e_config: E2EConfig) -> E2EConfig:
    """Config with resolved base URL (primary or fallback)."""
    resolved = getattr(driver, "e2e_base_url", e2e_config.base_url)
    return E2EConfig(
        base_url=resolved,
        email=e2e_config.email,
        password=e2e_config.password,
        headless=e2e_config.headless,
        allow_gen=e2e_config.allow_gen,
        implicit_wait=e2e_config.implicit_wait,
        explicit_wait=e2e_config.explicit_wait,
        page_load_timeout=e2e_config.page_load_timeout,
    )


@pytest.fixture
def home(driver, config) -> HomePage:
    return HomePage(driver, config)


@pytest.fixture
def auth(driver, config) -> AuthPage:
    return AuthPage(driver, config)


@pytest.fixture
def sidebar(driver, config) -> SidebarPage:
    return SidebarPage(driver, config)


@pytest.fixture
def chat(driver, config) -> ChatPage:
    return ChatPage(driver, config)


@pytest.fixture
def studio(driver, config) -> StudioPage:
    return StudioPage(driver, config)


@pytest.fixture
def guest_ready(home: HomePage):
    """Navigate as guest with consent accepted and overlays dismissed."""
    home.prepare_guest_session()
    return home


@pytest.fixture
def logged_in(guest_ready: HomePage, auth: AuthPage, config: E2EConfig, home: HomePage):
    """Guest session + login with E2E_EMAIL / E2E_PASSWORD."""
    home.open_login()
    auth.login(config.email, config.password)
    # Logged-in users may see a consent modal if user_consent is false
    home.accept_guest_consent_if_present()
    home.dismiss_overlays()
    assert home.is_logged_in(), "Expected Logout control after successful login"
    return home
