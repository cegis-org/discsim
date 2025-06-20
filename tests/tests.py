import pytest
from playwright.sync_api import sync_playwright, expect
import os
import re

DEV_URL = "http://localhost:8501"

@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        yield browser
        browser.close()

def test_homepage(browser):
    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)
    iframe_element = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_element.wait_for(state="attached")
    assert iframe_element.count() > 0, "Iframe not found"
    frame = iframe_element.content_frame
    assert frame is not None, "Content frame not available"
    frame.get_by_role("link", name="Home").click()

    # Check if the Home Page appears on the main page
    expect(page.get_by_role("heading", name="Welcome to VALIData")).to_be_visible()

    context.close()


def test_third_party_sampling(browser):
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()
    page.goto(DEV_URL)
    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "Iframe not available"

    iframe.get_by_role("link", name="Intervention Design").click()
    submit_btn = page.get_by_test_id("stBaseButton-secondaryFormSubmit")
    submit_btn.wait_for(state="visible")
    submit_btn.click()

    page.wait_for_load_state()
    expect(page.get_by_role("img", name="0").first).to_be_visible()
    expect(page.get_by_role("img", name="0").nth(1)).to_be_visible()

    context.close()

def test_intervention_analytics(browser):
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Intervention Analytics").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        print(options)
        options[1].click()
        # first_option_text = options[1].inner_text()
        # print(f"✅ Selected first option: {first_option_text}")
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()
    awc_tab = page.get_by_role("tab", name="AWC Level")
    awc_tab.wait_for()
    awc_tab.click()
    page.wait_for_load_state()

    expect(page.locator("summary").filter(has_text=re.compile(r"🔴 Red Zone.*"))).to_be_visible()
    expect(page.locator("summary").filter(has_text=re.compile(r"🟢 Green Zone.*"))).to_be_visible()
    expect(page.locator("summary").filter(has_text=re.compile(r"🟡 Yellow Zone.*"))).to_be_visible()

    context.close()

def test_check_admin_data_preview(browser):
    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Admin Data Diagnostic").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        page.wait_for_selector("[role='option']")
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        options[1].click()
        # first_option_text = options[1].inner_text()
        # print(f"✅ Selected first option: {first_option_text}")
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()
    #expect about the data page to be visible
    expect(page.get_by_role("heading", name="About the Data")).to_be_visible()

    context.close()

def test_identify_unique_ids(browser):

    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Admin Data Diagnostic").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        page.wait_for_selector("[role='option']")
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        options[1].click()
        # first_option_text = options[1].inner_text()
        # print(f"Selected first option: {first_option_text}")
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()

    page.get_by_role("button", name="Identify Unique ID(s)").click()
    page.get_by_test_id("stHorizontalBlock").get_by_test_id("stBaseButton-secondary").click()

    spinner = page.get_by_test_id("stSpinner").locator("div").first

    try:
        spinner.wait_for(state="detached", timeout=10000)
        print("Spinner completed")
    except:
        print("Spinner did not appear/disappear as expected")

    expect(page.get_by_test_id("stAlertContainer")).to_be_visible()

    context.close()

def test_verify_unique_ids(browser):
    
    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Admin Data Diagnostic").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        page.wait_for_selector("[role='option']")
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        options[1].click()
        # first_option_text = options[1].inner_text()
        # print(f"Selected first option: {first_option_text}")
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()
    page.get_by_role("button", name="Verify Unique ID(s)").click()


    page.get_by_test_id("stMultiSelect").locator("div").filter(has_text="Choose an option").nth(3).click()
    page.get_by_role("option", name="child").click()
    page.get_by_test_id("stMultiSelect").get_by_test_id("stWidgetLabel").click()
    page.get_by_test_id("stBaseButton-secondaryFormSubmit").click()

    spinner = page.get_by_test_id("stSpinner").locator("div").first

    try:
        spinner.wait_for(state="detached", timeout=10000)
        print("Spinner completed")
    except:
        print("Spinner did not appear/disappear as expected")

    expect(page.get_by_test_id("stAlertContainer")).to_be_visible()

    context.close()

def test_verify_unique_ids_create_unique_id(browser):
    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Admin Data Diagnostic").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        page.wait_for_selector("[role='option']")
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        options[1].click()
        # first_option_text = options[1].inner_text()
        # print(f"Selected first option: {first_option_text}")
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()
    page.get_by_role("button", name="Verify Unique ID(s)").click()


    page.get_by_test_id("stMultiSelect").locator("div").filter(has_text="Choose an option").nth(3).click()
    page.get_by_role("option", name="child").click()
    page.get_by_test_id("stMultiSelect").get_by_test_id("stWidgetLabel").click()
    page.get_by_test_id("stBaseButton-secondaryFormSubmit").click()

    spinner = page.get_by_test_id("stSpinner").locator("div").first

    try:
        spinner.wait_for(state="detached", timeout=10000)
        print("Spinner completed")
    except:
        print("Spinner did not appear/disappear as expected")

    # page.get_by_test_id("stAlertContainer").to_be_visible()

    page.get_by_text("Create New Unique ID Column").click()
    page.get_by_test_id("stExpanderDetails").get_by_test_id("stBaseButton-secondary").click()
    page.wait_for_load_state()
    expect(page.locator(".dvn-scroller")).to_be_visible()

    context.close()


def test_inspect_duplicate_entries(browser):
    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Admin Data Diagnostic").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        page.wait_for_selector("[role='option']")
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        options[1].click()
        # first_option_text = options[1].inner_text()
        # print(f"Selected first option: {first_option_text}")
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()
    page.get_by_role("button", name="Inspect Duplicate Entries").click()

    page.get_by_test_id("stMultiSelect").locator("div").filter(has_text="Choose an option").nth(3).click()
    page.get_by_role("option", name="child").click()
    page.get_by_test_id("stMainBlockContainer").click()
    page.get_by_test_id("stBaseButton-secondaryFormSubmit").click()

    spinner = page.get_by_test_id("stSpinner").locator("div").first

    try:
        spinner.wait_for(state="detached", timeout=10000)
        print("Spinner completed")
    except:
        print("Spinner did not appear/disappear as expected")

    page.wait_for_load_state()
    expect(page.get_by_text("No Duplicate Entries")).to_be_visible()

    context.close()


def test_inspect_duplicate_rows(browser):
    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Admin Data Diagnostic").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        page.wait_for_selector("[role='option']")
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        options[1].click()
        # first_option_text = options[1].inner_text()
        # print(f"Selected first option: {first_option_text}")
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()
    page.get_by_role("button", name="Inspect Duplicate Rows").click()

    page.get_by_test_id("stHorizontalBlock").get_by_test_id("stBaseButton-secondary").click()

    spinner = page.get_by_test_id("stSpinner").locator("div").first

    try:
        spinner.wait_for(state="detached", timeout=10000)
        print("Spinner completed")
    except:
        print("Spinner did not appear/disappear as expected")

    page.wait_for_load_state()
    expect(page.locator("summary")).to_be_visible()
    
    context.close()

def test_analyse_missing_entries(browser):
    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Admin Data Diagnostic").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        page.wait_for_selector("[role='option']")
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        options[1].click()
        # first_option_text = options[1].inner_text()
        # print(f"Selected first option: {first_option_text}")
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()
    page.get_by_test_id("stButtonGroup").get_by_role("button", name="Analyse Missing Entries").click()
    
    page.locator("div").filter(has_text=re.compile(r"^child$")).first.click()
    page.get_by_role("option", name="child").click()
    page.get_by_role("button", name="Analyze Missing Entries").click()

    spinner = page.get_by_test_id("stSpinner").locator("div").first

    try:
        spinner.wait_for(state="detached", timeout=10000)
        print("Spinner completed")
    except:
        print("Spinner did not appear/disappear as expected")

    page.wait_for_load_state()
    expect(page.get_by_role("alert").filter(has_text="Missing entries analysed for")).to_be_visible()

    context.close()


def test_analyse_missing_entries_grouped(browser):
    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Admin Data Diagnostic").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        page.wait_for_selector("[role='option']")
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        options[1].click()
        # first_option_text = options[1].inner_text()
        # print(f"Selected first option: {first_option_text}")
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()
    page.get_by_test_id("stButtonGroup").get_by_role("button", name="Analyse Missing Entries").click()
    
    page.locator("div").filter(has_text=re.compile(r"^child$")).first.click()
    page.get_by_role("option", name="L1_height").click()

    selectbox = page.locator("div").filter(has_text=re.compile(r"^None$")).first
    selectbox.click()
    dropdown = page.get_by_test_id("stSelectboxVirtualDropdown")
    dropdown.get_by_role("option", name="L0_id").click()

    page.get_by_role("button", name="Analyze Missing Entries").click()

    spinner = page.get_by_test_id("stSpinner").locator("div").first

    try:
        spinner.wait_for(state="detached", timeout=10000)
        print("Spinner completed")
    except:
        print("Spinner did not appear/disappear as expected")

    page.wait_for_load_state()
    expect(page.get_by_text("Results are grouped by column")).to_be_visible()

    context.close()

def test_analyse_missing_entries_filtered(browser):
    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Admin Data Diagnostic").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        page.wait_for_selector("[role='option']")
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        options[1].click()
        # first_option_text = options[1].inner_text()
        # print(f"Selected first option: {first_option_text}")
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()
    page.get_by_test_id("stButtonGroup").get_by_role("button", name="Analyse Missing Entries").click()
    
    page.locator("div").filter(has_text=re.compile(r"^child$")).first.click()
    page.get_by_role("option", name="L1_height").click()

    selectbox = page.locator("div").filter(has_text=re.compile(r"^None$")).first
    selectbox.click()
    dropdown = page.get_by_test_id("stSelectboxVirtualDropdown")
    dropdown.get_by_role("option", name="L0_id").click()

    selectbox2 = page.locator("div").filter(has_text=re.compile(r"^None$")).first
    selectbox2.click()
    dropdown2 = page.get_by_test_id("stSelectboxVirtualDropdown")
    dropdown2.get_by_role("option", name="child").click()

    page.get_by_role("button", name="Analyze Missing Entries").click()

    spinner = page.get_by_test_id("stSpinner").locator("div").first

    try:
        spinner.wait_for(state="detached", timeout=10000)
        print("Spinner completed")
    except:
        print("Spinner did not appear/disappear as expected")

    page.wait_for_load_state()
    expect(page.get_by_text("Results are filtered by")).to_be_visible()

    context.close()


def test_analyse_zero_entries(browser):
    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Admin Data Diagnostic").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        page.wait_for_selector("[role='option']")
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        options[1].click()
        # first_option_text = options[1].inner_text()
        # print(f"Selected first option: {first_option_text}")
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()
    page.get_by_test_id("stButtonGroup").get_by_role("button", name="Analyse Zero Entries").click()
    
    # page.locator("div").filter(has_text=re.compile(r"^L1_height$")).first.click()
    # page.get_by_role("option", name="L1_height").click()
    page.get_by_role("button", name="Analyze Zero Entries").click()

    spinner = page.get_by_test_id("stSpinner").locator("div").first

    try:
        spinner.wait_for(state="detached", timeout=10000)
        print("Spinner completed")
    except:
        print("Spinner did not appear/disappear as expected")

    page.wait_for_load_state()
    expect(page.get_by_role("alert").filter(has_text="Zero entries analysed for")).to_be_visible()

    context.close()


def test_analyse_zero_entries_grouped(browser):
    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Admin Data Diagnostic").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        page.wait_for_selector("[role='option']")
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        options[1].click()
        # first_option_text = options[1].inner_text()
        # print(f"Selected first option: {first_option_text}")
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()
    page.get_by_test_id("stButtonGroup").get_by_role("button", name="Analyse Zero Entries").click()
    
    # page.locator("div").filter(has_text=re.compile(r"^L1_height$")).first.click()
    # page.get_by_role("option", name="L1_height").click()

    selectbox = page.locator("div").filter(has_text=re.compile(r"^None$")).first
    selectbox.click()
    dropdown = page.get_by_test_id("stSelectboxVirtualDropdown")
    dropdown.get_by_role("option", name="child").click()

    page.get_by_role("button", name="Analyze Zero Entries").click()

    spinner = page.get_by_test_id("stSpinner").locator("div").first

    try:
        spinner.wait_for(state="detached", timeout=10000)
        print("Spinner completed")
    except:
        print("Spinner did not appear/disappear as expected")

    page.wait_for_load_state()
    expect(page.get_by_text("Results are grouped by column")).to_be_visible()

    context.close()

def test_analyse_zero_entries_filtered(browser):
    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Admin Data Diagnostic").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        page.wait_for_selector("[role='option']")
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        options[1].click()
        # first_option_text = options[1].inner_text()
        # print(f"Selected first option: {first_option_text}")
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()
    page.get_by_test_id("stButtonGroup").get_by_role("button", name="Analyse Zero Entries").click()
    
    # page.locator("div").filter(has_text=re.compile(r"^child$")).first.click()
    # page.get_by_role("option", name="L1_height").click()

    selectbox = page.locator("div").filter(has_text=re.compile(r"^None$")).first
    selectbox.click()
    dropdown = page.get_by_test_id("stSelectboxVirtualDropdown")
    dropdown.get_by_role("option", name="child").click()

    selectbox2 = page.locator("div").filter(has_text=re.compile(r"^None$")).first
    selectbox2.click()
    dropdown2 = page.get_by_test_id("stSelectboxVirtualDropdown")
    dropdown2.get_by_role("option", name="child").click()

    page.get_by_role("button", name="Analyze Zero Entries").click()

    spinner = page.get_by_test_id("stSpinner").locator("div").first

    try:
        spinner.wait_for(state="detached", timeout=10000)
        print("Spinner completed")
    except:
        print("Spinner did not appear/disappear as expected")

    page.wait_for_load_state()
    expect(page.get_by_text("Results are filtered by")).to_be_visible()

    context.close()

def test_generate_frequency_table(browser):
    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Admin Data Diagnostic").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        page.wait_for_selector("[role='option']")
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        options[1].click()
        # first_option_text = options[1].inner_text()
        # print(f"Selected first option: {first_option_text}")
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()
    page.get_by_test_id("stButtonGroup").get_by_role("button", name="Generate frequency table").click()
    
    # page.locator("div").filter(has_text=re.compile(r"^L1_height$")).first.click()
    # page.get_by_role("option", name="L1_height").click()
    page.get_by_test_id("stMainBlockContainer").get_by_role("button", name="Generate Frequency Table").click()

    spinner = page.get_by_test_id("stSpinner").locator("div").first

    try:
        spinner.wait_for(state="detached", timeout=10000)
        print("Spinner completed")
    except:
        print("Spinner did not appear/disappear as expected")

    page.wait_for_load_state()
    expect(page.get_by_role("alert").filter(has_text="Full frequency table")).to_be_visible()

    context.close()

def test_generate_frequency_table_sorted(browser):
    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Admin Data Diagnostic").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        page.wait_for_selector("[role='option']")
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        options[1].click()
        # first_option_text = options[1].inner_text()
        # print(f"Selected first option: {first_option_text}")
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()
    page.get_by_test_id("stButtonGroup").get_by_role("button", name="Generate frequency table").click()
    
    page.locator("div").filter(has_text=re.compile(r"^None$")).first.click()
    page.get_by_role("option", name="Ascending").click()

    page.get_by_test_id("stMainBlockContainer").get_by_role("button", name="Generate Frequency Table").click()

    spinner = page.get_by_test_id("stSpinner").locator("div").first

    try:
        spinner.wait_for(state="detached", timeout=10000)
        print("Spinner completed")
    except:
        print("Spinner did not appear/disappear as expected")

    page.wait_for_load_state()
    expect(page.get_by_role("alert").filter(has_text="Frequency table sorted by")).to_be_visible()

    context.close()

def test_generate_frequency_table_grouped(browser):
    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Admin Data Diagnostic").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        page.wait_for_selector("[role='option']")
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        options[1].click()
        # first_option_text = options[1].inner_text()
        # print(f"Selected first option: {first_option_text}")
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()
    page.get_by_test_id("stButtonGroup").get_by_role("button", name="Generate frequency table").click()
    
    page.locator("div").filter(has_text=re.compile(r"^None$")).first.click()
    page.get_by_role("option", name="Ascending").click()

    page.locator("div").filter(has_text=re.compile(r"^None$")).first.click()
    page.get_by_role("option", name="L1_height").click()

    page.get_by_test_id("stMainBlockContainer").get_by_role("button", name="Generate Frequency Table").click()

    spinner = page.get_by_test_id("stSpinner").locator("div").first

    try:
        spinner.wait_for(state="detached", timeout=10000)
        print("Spinner completed")
    except:
        print("Spinner did not appear/disappear as expected")

    page.wait_for_load_state()
    expect(page.get_by_role("alert").filter(has_text="Combined Frequency Table for")).to_be_visible()

    context.close()

def test_generate_frequency_table_filtered(browser):
    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Admin Data Diagnostic").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        page.wait_for_selector("[role='option']")
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        options[1].click()
        # first_option_text = options[1].inner_text()
        # print(f"Selected first option: {first_option_text}")
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()
    page.get_by_test_id("stButtonGroup").get_by_role("button", name="Generate frequency table").click()
    
    page.locator("div").filter(has_text=re.compile(r"^None$")).first.click()
    page.get_by_role("option", name="Ascending").click()

    page.locator("div").filter(has_text=re.compile(r"^None$")).first.click()
    page.get_by_role("option", name="L1_height").click()

    page.locator("div").filter(has_text=re.compile(r"^None$")).first.click()
    page.get_by_role("option", name="child").click()

    page.get_by_test_id("stMainBlockContainer").get_by_role("button", name="Generate Frequency Table").click()

    spinner = page.get_by_test_id("stSpinner").locator("div").first

    try:
        spinner.wait_for(state="detached", timeout=10000)
        print("Spinner completed")
    except:
        print("Spinner did not appear/disappear as expected")

    page.wait_for_load_state()
    expect(page.get_by_role("alert").filter(has_text="Results are filtered by child")).to_be_visible()

    context.close()

def test_analyse_data_quality(browser):
    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Admin Data Diagnostic").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        page.wait_for_selector("[role='option']")
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        options[1].click()
        # first_option_text = options[1].inner_text()
        # print(f"Selected first option: {first_option_text}")
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()
    page.get_by_test_id("stButtonGroup").get_by_role("button", name="Analyse the data quality of an indicator").click()

    page.get_by_test_id("stMainBlockContainer").get_by_role("button", name="Analyse the indicator values").click()

    spinner = page.get_by_test_id("stSpinner").locator("div").first

    try:
        spinner.wait_for(state="detached", timeout=10000)
        print("Spinner completed")
    except:
        print("Spinner did not appear/disappear as expected")

    page.wait_for_load_state()
    expect(page.get_by_role("alert").filter(has_text="Indicator Fill Rate Result:")).to_be_visible()

    context.close()

def test_analyse_data_quality_string(browser):
    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Admin Data Diagnostic").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        page.wait_for_selector("[role='option']")
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        options[1].click()
        # first_option_text = options[1].inner_text()
        # print(f"Selected first option: {first_option_text}")
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()
    page.get_by_test_id("stButtonGroup").get_by_role("button", name="Analyse the data quality of an indicator").click()

    page.get_by_test_id("stNumberInputStepUp").click()
    page.wait_for_load_state()
    page.get_by_role("textbox", name="Enter value").click()
    page.get_by_role("textbox", name="Enter value").fill("a")

    page.get_by_test_id("stNumberInputStepUp").click()
    page.wait_for_load_state()
    page.locator("#text_input_12").click()
    page.locator("#text_input_12").fill("b")
    
    page.get_by_test_id("stNumberInputStepUp").click()
    page.wait_for_load_state()
    page.locator("#text_input_16").click()
    page.locator("#text_input_16").fill("c")

    page.get_by_test_id("stMainBlockContainer").get_by_role("button", name="Analyse the indicator values").click()

    spinner = page.get_by_test_id("stSpinner").locator("div").first

    try:
        spinner.wait_for(state="detached", timeout=10000)
        print("Spinner completed")
    except:
        print("Spinner did not appear/disappear as expected")

    page.wait_for_load_state()
    expect(page.get_by_role("alert").filter(has_text="Custom invalid conditions")).to_be_visible()

    context.close()

def test_analyse_data_quality_string_grouped(browser):
    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Admin Data Diagnostic").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        page.wait_for_selector("[role='option']")
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        options[1].click()
        # first_option_text = options[1].inner_text()
        # print(f"Selected first option: {first_option_text}")
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()

    page.get_by_test_id("stButtonGroup").get_by_role("button", name="Analyse the data quality of an indicator").click()

    page.locator("div").filter(has_text=re.compile(r"^None$")).first.click()
    page.get_by_role("option", name="L1_height").click()

    page.get_by_test_id("stNumberInputStepUp").click()
    page.wait_for_load_state()
    page.locator("#text_input_8").click()
    page.locator("#text_input_8").fill("a")

    page.get_by_test_id("stNumberInputStepUp").click()
    page.wait_for_load_state()
    page.locator("#text_input_12").click()
    page.locator("#text_input_12").fill("b")
    
    page.get_by_test_id("stNumberInputStepUp").click()
    page.wait_for_load_state()
    page.locator("#text_input_16").click()
    page.locator("#text_input_16").fill("c")

    page.get_by_test_id("stMainBlockContainer").get_by_role("button", name="Analyse the indicator values").click()

    spinner = page.get_by_test_id("stSpinner").locator("div").first

    try:
        spinner.wait_for(state="detached", timeout=10000)
        print("Spinner completed")
    except:
        print("Spinner did not appear/disappear as expected")

    page.wait_for_load_state()
    expect(page.get_by_role("alert").filter(has_text="Indicator Fill Rate Report by Group:")).to_be_visible()

    context.close()


def test_analyse_data_quality_string_filtered(browser):
    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Admin Data Diagnostic").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        page.wait_for_selector("[role='option']")
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        options[1].click()
        # first_option_text = options[1].inner_text()
        # print(f"Selected first option: {first_option_text}")
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()

    page.get_by_test_id("stButtonGroup").get_by_role("button", name="Analyse the data quality of an indicator").click()

    page.locator("div").filter(has_text=re.compile(r"^None$")).first.click()
    page.get_by_role("option", name="L1_height").click()
    
    page.locator("div").filter(has_text=re.compile(r"^None$")).first.click()
    page.get_by_role("option", name="child").click()
    page.wait_for_load_state()

    page.get_by_test_id("stNumberInputStepUp").click()
    page.wait_for_load_state()
    page.locator("#text_input_8").click()
    page.locator("#text_input_8").fill("a")

    page.get_by_test_id("stNumberInputStepUp").click()
    page.wait_for_load_state()
    page.locator("#text_input_12").click()
    page.locator("#text_input_12").fill("b")
    
    page.get_by_test_id("stNumberInputStepUp").click()
    page.wait_for_load_state()
    page.locator("#text_input_16").click()
    page.locator("#text_input_16").fill("c")

    page.get_by_test_id("stMainBlockContainer").get_by_role("button", name="Analyse the indicator values").click()

    spinner = page.get_by_test_id("stSpinner").locator("div").first

    try:
        spinner.wait_for(state="detached", timeout=10000)
        print("Spinner completed")
    except:
        print("Spinner did not appear/disappear as expected")

    page.wait_for_load_state()
    expect(page.get_by_role("alert").filter(has_text="Results are filtered by child = a")).to_be_visible()

    context.close()

def test_analyse_data_quality_numeric(browser):
    context = browser.new_context()
    page = context.new_page()
    page.goto(DEV_URL)

    iframe_locator = page.locator("iframe[title=\"streamlit_navigation_bar\\.st_navbar\"]")
    iframe_locator.wait_for(state="attached")
    iframe = iframe_locator.first.content_frame
    assert iframe is not None, "❌ Iframe not found"

    iframe.get_by_role("link", name="Admin Data Diagnostic").click()
    page.get_by_text("Select a previously uploaded").wait_for()
    page.get_by_text("Select a previously uploaded").click()

    selectbox = page.get_by_test_id("stSelectbox").locator("div").filter(has_text="Select a file").nth(2)
    selectbox.wait_for()

    try:
        selectbox.click()
        page.wait_for_selector("[role='option']")
        options = page.get_by_role("option").all()
        assert options, "❌ No options found in the dropdown"
        options[1].click()
    except Exception as e:
        print(f"❌ Error while interacting with the dropdown: {e}")
        assert False

    page.wait_for_load_state()
    page.get_by_test_id("stButtonGroup").get_by_role("button", name="Analyse the data quality of an indicator").click()

    page.locator("div").filter(has_text=re.compile(r"^child$")).first.click()
    page.get_by_role("option", name="L1_height").click()

    page.get_by_test_id("stNumberInputStepUp").click()
    page.wait_for_load_state()
    page.get_by_role("spinbutton", name="Value").click()
    page.get_by_role("spinbutton", name="Value").fill("90")
    page.get_by_role("spinbutton", name="Value").press("Enter")

    page.get_by_test_id("stMainBlockContainer").get_by_role("button", name="Analyse the indicator values").click()

    spinner = page.get_by_test_id("stSpinner").locator("div").first

    try:
        spinner.wait_for(state="detached", timeout=10000)
        print("Spinner completed")
    except:
        print("Spinner did not appear/disappear as expected")

    page.wait_for_load_state()
    expect(page.get_by_role("alert").filter(has_text="Custom invalid conditions")).to_be_visible()

    context.close()