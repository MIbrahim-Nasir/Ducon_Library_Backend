"""Shared Selenium helpers for Ducon E2E tests."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

Locator = Tuple[str, str]


class TurnstileBlocked(Exception):
    """Raised when Cloudflare Turnstile / challenge blocks automation."""


def wait(driver: WebDriver, timeout: float) -> WebDriverWait:
    return WebDriverWait(
        driver,
        timeout,
        ignored_exceptions=(StaleElementReferenceException,),
    )


def find(
    driver: WebDriver,
    locators: Union[Locator, Sequence[Locator]],
    timeout: float = 25,
    *,
    clickable: bool = False,
) -> WebElement:
    if isinstance(locators, tuple) and len(locators) == 2 and isinstance(locators[0], str):
        candidates: Sequence[Locator] = [locators]  # type: ignore[list-item]
    else:
        candidates = locators  # type: ignore[assignment]

    last_err: Optional[Exception] = None
    for by, value in candidates:
        try:
            condition = EC.element_to_be_clickable((by, value)) if clickable else EC.presence_of_element_located((by, value))
            return wait(driver, timeout).until(condition)
        except TimeoutException as exc:
            last_err = exc
    raise TimeoutException(f"None of the locators matched within {timeout}s: {candidates}") from last_err


def find_optional(
    driver: WebDriver,
    locators: Union[Locator, Sequence[Locator]],
    timeout: float = 3,
) -> Optional[WebElement]:
    try:
        return find(driver, locators, timeout=timeout)
    except TimeoutException:
        return None


def click(
    driver: WebDriver,
    locators: Union[Locator, Sequence[Locator]],
    timeout: float = 25,
) -> WebElement:
    el = find(driver, locators, timeout=timeout, clickable=True)
    try:
        el.click()
    except Exception:
        driver.execute_script("arguments[0].click();", el)
    return el


def visible_text(driver: WebDriver, text: str, timeout: float = 25) -> WebElement:
    return find(
        driver,
        (By.XPATH, f"//*[contains(normalize-space(.), {xpath_literal(text)})]"),
        timeout=timeout,
    )


def xpath_literal(s: str) -> str:
    if "'" not in s:
        return f"'{s}'"
    if '"' not in s:
        return f'"{s}"'
    parts = s.split("'")
    return "concat(" + ", \"'\", ".join(f"'{p}'" for p in parts) + ")"


def is_turnstile_blocking(driver: WebDriver) -> bool:
    """Detect Cloudflare Turnstile / challenge overlays that block automation."""
    markers: Iterable[Locator] = (
        (By.CSS_SELECTOR, "iframe[src*='challenges.cloudflare.com']"),
        (By.CSS_SELECTOR, "iframe[src*='turnstile']"),
        (By.CSS_SELECTOR, ".cf-turnstile"),
        (By.CSS_SELECTOR, "#challenge-form"),
        (By.CSS_SELECTOR, "[data-cf-turnstile-response]"),
        (By.XPATH, "//*[contains(., 'Verify you are human') or contains(., 'Checking your browser')]"),
    )
    for by, value in markers:
        try:
            els = driver.find_elements(by, value)
            for el in els:
                try:
                    if el.is_displayed():
                        return True
                except StaleElementReferenceException:
                    continue
        except NoSuchElementException:
            continue
    # Title / body heuristics for interstitial challenge pages
    title = (driver.title or "").lower()
    if "just a moment" in title or "attention required" in title:
        return True
    return False


def skip_if_turnstile(driver: WebDriver, pytest_module) -> None:
    if is_turnstile_blocking(driver):
        pytest_module.skip(
            "Cloudflare Turnstile / challenge is blocking automation. "
            "Complete the challenge manually in a headed browser, or run with a "
            "Turnstile test key / allowlisted IP."
        )


def js_set_local_storage(driver: WebDriver, key: str, value: str) -> None:
    driver.execute_script(
        "localStorage.setItem(arguments[0], arguments[1]);",
        key,
        value,
    )


def dismiss_overlays(driver: WebDriver, timeout: float = 3) -> None:
    """Best-effort dismiss onboarding tips / non-mandatory overlays."""
    for label in ("Skip", "OK", "Got it", "Close"):
        el = find_optional(
            driver,
            (
                (By.XPATH, f"//button[normalize-space()={xpath_literal(label)}]"),
                (By.CSS_SELECTOR, f"button[aria-label={xpath_literal(label)}]"),
            ),
            timeout=timeout if label == "Skip" else 1,
        )
        if el is not None:
            try:
                if el.is_displayed():
                    driver.execute_script("arguments[0].click();", el)
            except Exception:
                pass
