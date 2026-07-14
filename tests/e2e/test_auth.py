"""Authenticated E2E flows."""

from __future__ import annotations

from pathlib import Path

import pytest
from selenium.webdriver.common.by import By

from tests.e2e import helpers

pytestmark = pytest.mark.e2e


def test_login_and_logged_in_chrome(logged_in, home, driver, config):
    helpers.skip_if_turnstile(driver, pytest)
    assert home.is_logged_in()
    # Saved bookmarks shortcut appears for authenticated users
    assert helpers.find_optional(driver, home.SAVED_BTN, timeout=5) is not None


def test_auth_ai_generations_panel(logged_in, home, sidebar, driver):
    helpers.skip_if_turnstile(driver, pytest)
    home.open_sidebar()
    sidebar.open_ai_generations()
    sidebar.assert_ai_generations_panel(expect_guest_gate=False)


def test_auth_uploads(logged_in, home, sidebar, sample_upload_png: Path, driver):
    helpers.skip_if_turnstile(driver, pytest)
    home.open_sidebar()
    sidebar.open_uploads()
    sidebar.upload_fixture(sample_upload_png)


def test_auth_chat_ui(logged_in, chat, driver, config):
    helpers.skip_if_turnstile(driver, pytest)
    chat.open()
    chat.assert_ui_loaded()
    if config.allow_gen:
        chat.send_message("E2E authenticated smoke — do not generate images.", wait_for_response=False)


def test_auth_studio_smoke(logged_in, studio, driver):
    helpers.skip_if_turnstile(driver, pytest)
    studio.open_studio()
    studio.assert_steps_visible()
    studio.assert_upload_step()
    studio.close()


def test_auth_catalog_and_search(logged_in, home, driver, config):
    helpers.skip_if_turnstile(driver, pytest)
    home.browse_catalog_tabs()
    home.open_search()
    # Search UI varies; assert search control was activatable
    helpers.find(
        driver,
        (
            (By.CSS_SELECTOR, "input[type='search'], input[placeholder*='Search'], input[aria-label*='Search']"),
            (By.XPATH, "//input[contains(@placeholder,'Search') or contains(@aria-label,'Search')]"),
        ),
        timeout=config.explicit_wait,
    )


def test_logout(logged_in, home, driver):
    helpers.skip_if_turnstile(driver, pytest)
    home.logout()
    assert not home.is_logged_in()
