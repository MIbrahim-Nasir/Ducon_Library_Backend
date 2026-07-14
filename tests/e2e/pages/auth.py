"""Auth modal page object."""

from __future__ import annotations

from selenium.webdriver.common.by import By

from tests.e2e import helpers
from tests.e2e.pages.base import BasePage


class AuthPage(BasePage):
    DIALOG = (
        (By.CSS_SELECTOR, "[role='dialog'][aria-label='Login']"),
        (By.CSS_SELECTOR, ".auth-modal-panel"),
        (By.XPATH, "//*[@role='dialog' and (@aria-label='Login' or @aria-label='Create account')]"),
    )
    EMAIL = (
        (By.CSS_SELECTOR, ".auth-modal-panel input[type='email']"),
        (By.CSS_SELECTOR, "input[type='email'][placeholder='name@gmail.com']"),
        (By.CSS_SELECTOR, "input[type='email']"),
    )
    PASSWORD = (
        (By.CSS_SELECTOR, ".auth-modal-panel input[type='password']"),
        (By.CSS_SELECTOR, "input[type='password']"),
    )
    SUBMIT = (
        (By.XPATH, "//div[contains(@class,'auth-modal') or @role='dialog']//button[@type='submit' and contains(., 'Login')]"),
        (By.XPATH, "//button[@type='submit' and normalize-space()='Login']"),
    )
    WELCOME = (
        (By.XPATH, "//h2[contains(., 'Welcome Back')]"),
    )
    ERROR = (
        (By.CSS_SELECTOR, ".auth-form [style*='ef4444'], .auth-form div"),
    )

    def wait_open(self) -> None:
        helpers.find(self.driver, self.DIALOG, timeout=self.timeout)
        helpers.find(self.driver, self.WELCOME, timeout=self.timeout)

    def login(self, email: str, password: str) -> None:
        self.wait_open()
        email_el = helpers.find(self.driver, self.EMAIL, timeout=self.timeout)
        email_el.clear()
        email_el.send_keys(email)
        pwd_el = helpers.find(self.driver, self.PASSWORD, timeout=self.timeout)
        pwd_el.clear()
        pwd_el.send_keys(password)
        helpers.click(self.driver, self.SUBMIT, timeout=self.timeout)
