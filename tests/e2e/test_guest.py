"""Guest-mode UI smoke tests (no paid AI generations by default)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e import helpers

pytestmark = pytest.mark.e2e


def test_guest_lands_and_sees_hero(guest_ready, home, driver, config):
    """Guest can load the app past consent and see the hero."""
    helpers.skip_if_turnstile(driver, pytest)
    assert "ducon" in config.base_url.lower()
    helpers.visible_text(driver, "Your space", timeout=config.explicit_wait)
    assert helpers.find_optional(driver, home.LOGIN_BTN, timeout=5) is not None


def test_guest_catalog_browse(guest_ready, home, driver):
    helpers.skip_if_turnstile(driver, pytest)
    home.scroll_to_catalog()
    home.browse_catalog_tabs()


def test_guest_bookmark_button(guest_ready, home, driver):
    helpers.skip_if_turnstile(driver, pytest)
    home.scroll_to_catalog()
    home.bookmark_first_design()


def test_guest_open_uploads_and_upload_fixture(guest_ready, home, sidebar, sample_upload_png: Path, driver):
    helpers.skip_if_turnstile(driver, pytest)
    home.open_sidebar()
    sidebar.open_uploads()
    sidebar.upload_fixture(sample_upload_png)


def test_guest_ai_generations_gate(guest_ready, home, sidebar, driver):
    """Guests see the sign-in gate for AI Generations (no gen triggered)."""
    helpers.skip_if_turnstile(driver, pytest)
    home.open_sidebar()
    sidebar.open_ai_generations()
    sidebar.assert_ai_generations_panel(expect_guest_gate=True)


def test_guest_open_bookmarks_panel(guest_ready, home, sidebar, driver):
    helpers.skip_if_turnstile(driver, pytest)
    home.open_sidebar()
    sidebar.open_bookmarks()


def test_guest_chat_ui_loads(guest_ready, chat, driver, config):
    """Open chat and verify composer; send only if E2E_ALLOW_GEN=1."""
    helpers.skip_if_turnstile(driver, pytest)
    chat.open()
    chat.assert_ui_loaded()
    if config.allow_gen:
        chat.send_message("Hello from E2E smoke — reply briefly.", wait_for_response=False)
    else:
        # Type without sending to avoid burning guest chat quota / Turnstile
        from selenium.webdriver.common.by import By

        box = helpers.find(driver, chat.MESSAGE_INPUT, timeout=config.explicit_wait)
        box.click()
        box.send_keys("E2E UI check (not sent)")


def test_guest_studio_opens_steps(guest_ready, studio, driver):
    """Open studio wizard and verify stepper / upload step — do not visualize."""
    helpers.skip_if_turnstile(driver, pytest)
    studio.open_studio()
    studio.assert_steps_visible()
    studio.assert_upload_step()
    studio.close()


@pytest.mark.e2e_gen
def test_guest_gen_gated(guest_ready, config):
    """Placeholder: real guest generation only when E2E_ALLOW_GEN=1."""
    if not config.allow_gen:
        pytest.skip("Set E2E_ALLOW_GEN=1 to run generation-touching tests on production")
    pytest.skip("No automatic paid generation path implemented — use UI smoke tests instead")
