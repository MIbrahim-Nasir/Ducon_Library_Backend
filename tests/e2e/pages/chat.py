"""Designer chat page object."""

from __future__ import annotations

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from tests.e2e import helpers
from tests.e2e.pages.base import BasePage


class ChatPage(BasePage):
    OPEN_BTN = (
        (By.CSS_SELECTOR, "button[aria-label='Open Ducon Designer chat']"),
        (By.XPATH, "//button[@aria-label='Open Ducon Designer chat']"),
    )
    MESSAGE_INPUT = (
        (By.CSS_SELECTOR, "textarea[aria-label='Message input'], input[aria-label='Message input']"),
        (By.CSS_SELECTOR, "[aria-label='Message input']"),
        (By.XPATH, "//*[@aria-label='Message input']"),
    )
    SEND_BTN = (
        (By.CSS_SELECTOR, "button[aria-label='Send message']"),
        (By.XPATH, "//button[@aria-label='Send message']"),
    )
    CLOSE_BTN = (
        (By.CSS_SELECTOR, "button[aria-label='Close chat']"),
        (By.XPATH, "//button[@aria-label='Close chat' or @aria-label='Minimize chat']"),
    )
    ATTACH_BTN = (
        (By.CSS_SELECTOR, "button[aria-label='Attach image']"),
    )
    TOOL_UI = (
        (By.CSS_SELECTOR, "[class*='ck-job'], [class*='dc-tool'], [class*='tool']"),
        (By.XPATH, "//*[contains(@class,'ck-job') or contains(@class,'dc-tool')]"),
    )

    def open(self) -> None:  # noqa: A003 — page action name
        helpers.click(self.driver, self.OPEN_BTN, timeout=self.timeout)
        helpers.find(self.driver, self.MESSAGE_INPUT, timeout=self.timeout)

    def send_message(self, text: str, *, wait_for_response: bool = False) -> None:
        """Type a message. By default does not wait for a paid AI reply."""
        box = helpers.find(self.driver, self.MESSAGE_INPUT, timeout=self.timeout)
        box.click()
        box.clear()
        box.send_keys(text)
        send = helpers.find_optional(self.driver, self.SEND_BTN, timeout=3)
        if send is not None and send.is_enabled():
            helpers.click(self.driver, self.SEND_BTN, timeout=self.timeout)
        else:
            box.send_keys(Keys.ENTER)
        if wait_for_response:
            # Optional: only when E2E_ALLOW_GEN=1
            helpers.find(
                self.driver,
                (
                    (By.XPATH, "//*[contains(@class,'dc-msg') or contains(@class,'ck-')]"),
                ),
                timeout=self.timeout,
            )

    def assert_ui_loaded(self) -> None:
        helpers.find(self.driver, self.MESSAGE_INPUT, timeout=self.timeout)
        # Attach / visualize controls are nice-to-have
        helpers.find_optional(self.driver, self.ATTACH_BTN, timeout=2)

    def close(self) -> None:
        btn = helpers.find_optional(self.driver, self.CLOSE_BTN, timeout=3)
        if btn is not None:
            self.driver.execute_script("arguments[0].click();", btn)
