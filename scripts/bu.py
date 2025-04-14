import asyncio

from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.dom.service import DomService


async def debug_clickable_elements(url: str):
    # Initialize browser with visible window (not headless) for debugging
    config = BrowserContextConfig(
        disable_security=True,
        wait_for_network_idle_page_load_time=2,
    )

    browser = Browser(
        config=BrowserConfig(
            chrome_instance_path="/usr/bin/google-chrome-stable", headless=False
        )
    )
    context = BrowserContext(browser=browser, config=config)

    async with context as context:
        page = await context.get_current_page()
        dom_service = DomService(page)

        # Navigate to the page
        print(f"\nNavigating to {url}")
        await page.goto(url)
        await asyncio.sleep(2)  # Wait for page to stabilize

        try:
            # Get all clickable elements with highlighting
            print("\nGetting all clickable elements:")
            dom_state = await dom_service.get_clickable_elements(
                highlight_elements=True,
                viewport_expansion=1000,  # Expand viewport to see more elements
            )

            # Print all clickable elements with their properties
            print("\nClickable Elements Found:")
            print("-" * 50)
            elements_string = dom_state.element_tree.clickable_elements_to_string()
            print(elements_string)

            # Print detailed information about each element in selector map
            print("\nDetailed Element Information:")
            print("-" * 50)
            for index, element in dom_state.selector_map.items():
                print(f"\nElement [{index}]:")
                print(f"Tag: {element.tag_name}")
                print(f"XPath: {element.xpath}")
                print(f"Attributes: {element.attributes}")
                print(f"Is Interactive: {element.is_interactive}")
                print(f"Is Visible: {element.is_visible}")
                # print(f"Is In Viewport: {element.is_in_viewport}")
                # if element.viewport_coordinates:
                #    print(f"Viewport Coordinates: {element.viewport_coordinates}")

            # Keep the browser window open for inspection
            input("\nPress Enter to clear highlights...")
            await page.evaluate(
                'document.getElementById("playwright-highlight-container")?.remove()'
            )

            # Keep window open for manual inspection
            input("\nPress Enter to close the browser...")

        except Exception as e:
            print(f"Error during debugging: {e}")


if __name__ == "__main__":
    # Replace with your target URL
    target_url = "https://www.linkedin.com/jobs/view/4199448826"
    asyncio.run(debug_clickable_elements(target_url))
