"""Studio wizard page object — smoke only (no paid visualize)."""

from __future__ import annotations

from selenium.webdriver.common.by import By

from tests.e2e import helpers
from tests.e2e.pages.base import BasePage


class StudioPage(BasePage):
    OPEN_BTNS = (
        (By.XPATH, "//button[contains(., 'Try it with your photo')]"),
        (By.CSS_SELECTOR, "button[aria-label='Visualize a design in your space']"),
        (By.XPATH, "//button[contains(., 'See in my space')]"),
    )
    TITLE = (
        (By.XPATH, "//*[contains(@class,'sw-header-title') and contains(., 'Visualize my space')]"),
        (By.XPATH, "//*[normalize-space()='Visualize my space']"),
    )
    STEPPER = (
        (By.CSS_SELECTOR, "nav[aria-label='Wizard progress']"),
        (By.XPATH, "//nav[@aria-label='Wizard progress']"),
    )
    STEP_PHOTO = (
        (By.XPATH, "//button[@aria-label='Go to Photo step'] | //*[contains(@class,'sw-stepper') and contains(., 'Photo')]"),
    )
    STEP_LABELS = ("Photo", "Space", "Style", "Products", "Directions", "Visualize")
    CLOSE = (
        (By.CSS_SELECTOR, ".sw-close, button[aria-label='Close']"),
        (By.XPATH, "//button[contains(@class,'sw-close') or @aria-label='Close']"),
    )
    UPLOAD_UI = (
        (By.XPATH, "//*[contains(., 'Drop your photos') or contains(., 'Upload or pick') or contains(., 'Good shot')]"),
        (By.CSS_SELECTOR, ".fu-root, .sw-photo-guide, input[type='file']"),
    )
    CONTINUE_OR_NEXT = (
        (By.XPATH, "//button[contains(., 'Continue') or contains(., 'Next')]"),
    )

    def open_studio(self) -> None:
        helpers.click(self.driver, self.OPEN_BTNS, timeout=self.timeout)
        helpers.find(self.driver, self.TITLE, timeout=self.timeout)
        helpers.find(self.driver, self.STEPPER, timeout=self.timeout)

    def assert_steps_visible(self) -> None:
        helpers.find(self.driver, self.STEPPER, timeout=self.timeout)
        for label in self.STEP_LABELS[:3]:
            helpers.visible_text(self.driver, label, timeout=self.timeout)

    def assert_upload_step(self) -> None:
        helpers.find(self.driver, self.UPLOAD_UI, timeout=self.timeout)

    def close(self) -> None:
        btn = helpers.find_optional(self.driver, self.CLOSE, timeout=3)
        if btn is not None:
            self.driver.execute_script("arguments[0].click();", btn)
