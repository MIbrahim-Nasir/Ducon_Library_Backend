"""Workspace sidebar + uploads / AI generations modals."""

from __future__ import annotations

from pathlib import Path

from selenium.webdriver.common.by import By

from tests.e2e import helpers
from tests.e2e.pages.base import BasePage


def _folder_locators(label: str):
    return (
        (By.XPATH, f"//div[contains(@class,'sidebar-folder')][.//span[normalize-space()='{label}']]"),
        (By.XPATH, f"//span[normalize-space()='{label}']/ancestor::div[contains(@class,'sidebar-folder')][1]"),
        (By.XPATH, f"//*[normalize-space()='{label}']"),
    )


class SidebarPage(BasePage):
    UPLOADS_TITLE = (
        (By.XPATH, "//*[contains(., 'Your Uploads') or contains(., 'Click or Drag')]"),
        (By.XPATH, "//h2[contains(., 'Your Uploads')] | //h3[contains(., 'Click or Drag')]"),
    )
    UPLOAD_DROPZONE_TEXT = (
        (By.XPATH, "//*[contains(., 'Click or Drag & Drop to Upload')]"),
    )
    FILE_INPUT = (
        (By.CSS_SELECTOR, "input[type='file'][accept*='image']"),
        (By.CSS_SELECTOR, "input[type='file']"),
    )
    AI_GEN_TITLE = (
        (By.XPATH, "//h2[contains(., 'AI Generations')]"),
        (By.XPATH, "//*[normalize-space()='AI Generations']"),
    )
    AI_GEN_GUEST_GATE = (
        (By.XPATH, "//*[contains(., 'Sign in to use AI Generation')]"),
    )
    AI_GEN_LIST_OR_EMPTY = (
        (By.XPATH, "//*[contains(., 'AI Generations') or contains(., 'No generations') or contains(., 'Create') or contains(., 'Sign in to use AI')]"),
    )
    BOOKMARKS_TITLE = (
        (By.XPATH, "//h2[contains(., 'All Bookmarks') or contains(., 'Bookmarks')]"),
        (By.XPATH, "//*[normalize-space()='All Bookmarks']"),
    )
    CLOSE_MODAL = (
        (By.CSS_SELECTOR, "button[aria-label='Close']"),
        (By.XPATH, "//button[@aria-label='Close']"),
    )

    def open_folder(self, label: str) -> None:
        helpers.click(self.driver, _folder_locators(label), timeout=self.timeout)

    def open_uploads(self) -> None:
        self.open_folder("Uploads")
        helpers.find(self.driver, self.UPLOAD_DROPZONE_TEXT, timeout=self.timeout)

    def open_ai_generations(self) -> None:
        self.open_folder("AI Gen")
        helpers.find(self.driver, self.AI_GEN_TITLE, timeout=self.timeout)

    def open_bookmarks(self) -> None:
        self.open_folder("Bookmarks")
        helpers.find(self.driver, self.BOOKMARKS_TITLE, timeout=self.timeout)

    def upload_fixture(self, path: Path) -> None:
        path = Path(path)
        assert path.exists(), f"Fixture missing: {path}"
        inp = helpers.find(self.driver, self.FILE_INPUT, timeout=self.timeout)
        inp.send_keys(str(path.resolve()))
        # Local IndexedDB save — wait for count or thumbnail text
        helpers.find(
            self.driver,
            (
                (By.XPATH, "//*[contains(., 'YOUR UPLOADS') and contains(., '(')]"),
                (By.XPATH, "//*[contains(., 'YOUR UPLOADS (1)') or contains(., 'YOUR UPLOADS (')]"),
            ),
            timeout=self.timeout,
        )

    def assert_ai_generations_panel(self, *, expect_guest_gate: bool | None = None) -> None:
        helpers.find(self.driver, self.AI_GEN_LIST_OR_EMPTY, timeout=self.timeout)
        if expect_guest_gate is True:
            helpers.find(self.driver, self.AI_GEN_GUEST_GATE, timeout=self.timeout)
        elif expect_guest_gate is False:
            gate = helpers.find_optional(self.driver, self.AI_GEN_GUEST_GATE, timeout=2)
            assert gate is None, "Expected authenticated AI Generations panel, still seeing guest gate"
