"""Base page object."""

from __future__ import annotations

from selenium.webdriver.remote.webdriver import WebDriver

from tests.e2e.config import E2EConfig
from tests.e2e import helpers


class BasePage:
    def __init__(self, driver: WebDriver, config: E2EConfig):
        self.driver = driver
        self.config = config
        self.timeout = config.explicit_wait

    def open(self, path: str = "/") -> None:
        url = f"{self.config.base_url}{path}"
        self.driver.get(url)

    def skip_if_turnstile(self, pytest_module) -> None:
        helpers.skip_if_turnstile(self.driver, pytest_module)

    def dismiss_overlays(self) -> None:
        helpers.dismiss_overlays(self.driver)
