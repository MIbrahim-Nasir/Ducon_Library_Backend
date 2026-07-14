"""Home / catalog / consent page object."""

from __future__ import annotations

from selenium.webdriver.common.by import By

from tests.e2e import helpers
from tests.e2e.pages.base import BasePage


class HomePage(BasePage):
    LOGIN_BTN = (
        (By.CSS_SELECTOR, "button.nav-login-btn"),
        (By.CSS_SELECTOR, "button[title='Login or sign up']"),
        (By.XPATH, "//button[normalize-space()='Login']"),
    )
    SIDEBAR_BTN = (
        (By.CSS_SELECTOR, "button[title='Open Sidebar']"),
        (By.XPATH, "//button[@title='Open Sidebar']"),
    )
    SEARCH_BTN = (
        (By.CSS_SELECTOR, "button[aria-label='Search designs']"),
        (By.XPATH, "//button[@aria-label='Search designs' or @title='Search']"),
    )
    HERO_HEADING = (
        (By.XPATH, "//h1[contains(., 'Your space') or contains(., 'See it before')]"),
    )
    TRY_PHOTO_BTN = (
        (By.XPATH, "//button[contains(., 'Try it with your photo')]"),
        (By.CSS_SELECTOR, "button[aria-label='Visualize a design in your space']"),
    )
    BROWSE_DESIGNS = (
        (By.XPATH, "//button[normalize-space()='Browse designs']"),
    )
    DESIGNS_TAB = (
        (By.XPATH, "//*[@role='tab' and (normalize-space()='Designs' or normalize-space()='DESIGNS')]"),
    )
    PRODUCTS_TAB = (
        (By.XPATH, "//*[@role='tab' and (normalize-space()='Products' or normalize-space()='PRODUCTS')]"),
    )
    AREAS_TAB = (
        (By.XPATH, "//*[@role='tab' and (normalize-space()='Areas' or normalize-space()='AREAS')]"),
    )
    BOOKMARK_BTN = (
        (By.CSS_SELECTOR, "button[aria-label='Bookmark this design']"),
        (By.XPATH, "//button[@aria-label='Bookmark this design']"),
    )
    LOGOUT_BTN = (
        (By.CSS_SELECTOR, "button[title^='Logout']"),
        (By.XPATH, "//button[starts-with(@title, 'Logout')]"),
    )
    SAVED_BTN = (
        (By.CSS_SELECTOR, "button[title='Saved']"),
        (By.XPATH, "//button[@title='Saved']"),
    )
    CONSENT_CHECKBOX = (
        (By.CSS_SELECTOR, "input[type='checkbox']"),
        (By.XPATH, "//label[contains(., 'I agree to Ducon')]/preceding::input[@type='checkbox'][1]"),
        (By.XPATH, "//*[contains(., 'I agree to Ducon Library')]/ancestor::label//input | //input[@type='checkbox']"),
    )
    CONSENT_CONTINUE = (
        (By.XPATH, "//button[contains(., 'Continue to Ducon Library')]"),
    )
    CHAT_OPEN = (
        (By.CSS_SELECTOR, "button[aria-label='Open Ducon Designer chat']"),
        (By.XPATH, "//button[@aria-label='Open Ducon Designer chat']"),
    )
    STUDIO_OPEN = (
        (By.CSS_SELECTOR, "button[aria-label='Visualize a design in your space']"),
        (By.XPATH, "//button[contains(., 'Try it with your photo')]"),
    )

    def accept_guest_consent_if_present(self) -> bool:
        """Accept sitewide guest consent wall if shown. Returns True if handled."""
        continue_btn = helpers.find_optional(self.driver, self.CONSENT_CONTINUE, timeout=5)
        if continue_btn is None:
            return False

        # Click the agreement checkbox (React controlled input).
        checkbox = helpers.find_optional(
            self.driver,
            (
                (By.XPATH, "//button[contains(., 'Continue to Ducon Library')]/ancestor::div[1]//input[@type='checkbox']"),
                (By.XPATH, "//label[.//span[contains(., 'I agree to Ducon Library')]]//input[@type='checkbox']"),
                (By.CSS_SELECTOR, "input[type='checkbox']"),
            ),
            timeout=3,
        )
        if checkbox is not None:
            try:
                if not checkbox.is_selected():
                    checkbox.click()
            except Exception:
                self.driver.execute_script(
                    "arguments[0].checked = true; arguments[0].dispatchEvent(new Event('change', {bubbles:true}));",
                    checkbox,
                )

        # Wait until Continue is enabled, then click
        helpers.wait(self.driver, self.timeout).until(
            lambda d: helpers.find(d, self.CONSENT_CONTINUE, timeout=2).is_enabled()
        )
        helpers.click(self.driver, self.CONSENT_CONTINUE, timeout=self.timeout)
        helpers.wait(self.driver, self.timeout).until(
            lambda d: helpers.find_optional(d, self.CONSENT_CONTINUE, timeout=1) is None
        )
        return True

    def seed_guest_consent(self) -> None:
        """Set localStorage consent so the wall does not block subsequent navigations."""
        helpers.js_set_local_storage(self.driver, "ducon_guest_consented", "true")

    def prepare_guest_session(self) -> None:
        """Land as guest: open app, accept consent, dismiss onboarding."""
        import pytest

        self.open("/")
        self.skip_if_turnstile(pytest)
        # Prefer real UI accept; also seed storage then refresh if wall still up
        accepted = self.accept_guest_consent_if_present()
        self.seed_guest_consent()
        if not accepted and helpers.find_optional(self.driver, self.CONSENT_CONTINUE, timeout=2):
            self.driver.refresh()
            self.accept_guest_consent_if_present()
            self.seed_guest_consent()
        self.dismiss_overlays()
        helpers.find(self.driver, self.HERO_HEADING, timeout=self.timeout)

    def open_login(self) -> None:
        helpers.click(self.driver, self.LOGIN_BTN, timeout=self.timeout)

    def open_sidebar(self) -> None:
        helpers.click(self.driver, self.SIDEBAR_BTN, timeout=self.timeout)
        helpers.visible_text(self.driver, "Workspace", timeout=self.timeout)

    def open_search(self) -> None:
        helpers.click(self.driver, self.SEARCH_BTN, timeout=self.timeout)

    def is_logged_in(self) -> bool:
        return helpers.find_optional(self.driver, self.LOGOUT_BTN, timeout=3) is not None

    def logout(self) -> None:
        helpers.click(self.driver, self.LOGOUT_BTN, timeout=self.timeout)
        helpers.find(self.driver, self.LOGIN_BTN, timeout=self.timeout)

    def browse_catalog_tabs(self) -> None:
        helpers.click(self.driver, self.PRODUCTS_TAB, timeout=self.timeout)
        helpers.click(self.driver, self.AREAS_TAB, timeout=self.timeout)
        helpers.click(self.driver, self.DESIGNS_TAB, timeout=self.timeout)

    def bookmark_first_design(self) -> None:
        helpers.click(self.driver, self.BOOKMARK_BTN, timeout=self.timeout)

    def scroll_to_catalog(self) -> None:
        btn = helpers.find_optional(self.driver, self.BROWSE_DESIGNS, timeout=3)
        if btn is not None:
            self.driver.execute_script("arguments[0].click();", btn)
        else:
            helpers.click(
                self.driver,
                (By.CSS_SELECTOR, "button[aria-label='Scroll to catalog']"),
                timeout=self.timeout,
            )
