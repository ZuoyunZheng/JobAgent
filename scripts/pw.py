from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
import asyncio
import os

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
            headless=False, slow_mo=500, user_data_dir=user_data_dir)
        page = await browser.new_page()
        await page.goto("https://www.linkedin.com/jobs/view/4199448826")
        #await page.goto("https://www.google.com")
        await page.wait_for_load_state("networkidle")
    
        # Find all potentially clickable elements
        clickable_elements = await page.query_selector_all(
            'a, button, input[type="button"], input[type="submit"], [role="button"], [onclick], [class*="btn"], [class*="button"]'
        )
       
        print(f"Found {len(clickable_elements)} potential clickable elements:")        
        # Keep browser open for manual inspection
        input("Press Enter to close the browser...")
        await browser.close()

if __name__=="__main__":
    asyncio.run(main())
