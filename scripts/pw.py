import asyncio
import os

from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("http://playwright.dev")
    print(page.title())
    browser.close()


async def main():
    async with async_playwright() as p:
        user_data_dir = os.path.expanduser("~/.config/google-chrome")
        browser = await p.chromium.launch_persistent_context(
            executable_path="/usr/bin/google-chrome-stable",
            headless=False,
            slow_mo=500,
            user_data_dir=user_data_dir,
        )
        page = await browser.new_page()
        await page.goto("https://www.linkedin.com/jobs/view/4199448826")
        # await page.goto("https://www.google.com")
        # await page.wait_for_load_state("networkidle")

        # Find all potentially clickable elements
        clickable_elements = await page.query_selector_all(
            'a, button, input[type="button"], input[type="submit"], [role="button"], [onclick], [class*="btn"], [class*="button"]'
        )

        print(f"Found {len(clickable_elements)} potential clickable elements:")

        for i, element in enumerate(clickable_elements):
            # Get useful properties
            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
            text = await element.text_content()
            text = text.strip() if text else "[No text]"

            # Get attributes
            href = await element.get_attribute("href") if tag_name == "a" else None
            id_attr = await element.get_attribute("id")
            classes = await element.get_attribute("class")

            # Check visibility
            is_visible = await element.is_visible()

            # Get bounding box if visible
            if is_visible:
                box = await element.bounding_box()
                position = (
                    f"at x:{box['x']:.0f}, y:{box['y']:.0f}"
                    if box
                    else "position unknown"
                )
            else:
                position = "not visible"

            # Print formatted info
            print(f"Element #{i}:")
            print(f"  Tag: <{tag_name}>")
            print(f"  Text: {text[:50]}{'...' if len(text) > 50 else ''}")
            print(f"  Visible: {is_visible} ({position})")
            if id_attr:
                print(f"  ID: {id_attr}")
            if classes:
                print(f"  Classes: {classes}")
            if href:
                print(f"  URL: {href}")
            print("---------------------------------------------------")

        # Keep browser open for manual inspection
        import pdb

        pdb.set_trace()
        input("Press Enter to close the browser...")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
