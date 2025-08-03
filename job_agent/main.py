import asyncio
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from .browser import main as browser_main
from .browser import main_with_jobs as browser_main_with_jobs
from .rag import load_retriever
from .scrape import jobspy_scrape_jobs
from .utils.args import parse_args


def main(args=None):
    if args is None:
        args = parse_args()
    load_dotenv()
    print("🤖 JobAgent - Automated Job Application System")
    print("=" * 50)

    retriever = None
    jobs = None

    # Initialize RAG pipeline if not skipped
    if not args.skip_rag:
        print("📚 Initializing RAG Pipeline...")
        retriever = load_retriever(args.reload_data)
        print("✅ RAG Pipeline Initialized.")
    else:
        print("⏭️  Skipping RAG Pipeline.")

    # Perform job scraping if not skipped
    if not args.skip_scraping:
        print(f"🔍 Scraping jobs for '{args.job_terms}' in {args.location}...")
        jobs = jobspy_scrape_jobs(
            ["linkedin"], args.job_terms, args.location, args.num_jobs, 72
        )
        print(f"✅ Job Scraping Completed. Found {len(jobs)} jobs.")

        # Save jobs to pickle file if requested
        if args.save_jobs:
            jobs.to_pickle("jobs.pkl")
            print(f"💾 Jobs saved to jobs.pkl")

        # Display job information
        if len(jobs) > 0:
            print("\n📋 Job Listings:")
            print("=" * 80)
            for i, (_, job) in enumerate(jobs.iterrows(), 1):
                print(f"{i}. {job.get('title', 'N/A')}")
                print(f"   Company: {job.get('company', 'N/A')}")
                print(f"   Location: {job.get('location', 'N/A')}")
                print(f"   Job URL: {job.get('job_url', 'N/A')}")
                # Show brief description preview
                description = str(job.get("description", ""))
                if description and description != "nan":
                    preview = description[:150].replace("\n", " ").strip()
                    print(f"   Preview: {preview}...")
                print("-" * 80)
        else:
            print("❌ No jobs found for the specified criteria.")
    else:
        print("⏭️  Skipping Job Scraping.")

    # Initialize Browser Automation if not skipped
    if not args.skip_browser:
        print("🌐 Starting Browser Automation...")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

        if jobs is not None and len(jobs) > 0:
            print(f"📋 Opening browser tabs for {len(jobs)} jobs...")
            job_urls = jobs["job_url"].dropna().tolist()
            if job_urls:
                asyncio.run(browser_main_with_jobs(llm, job_urls))
            else:
                print("⚠️  No valid job URLs found to open.")
        else:
            print(
                "⚠️  No jobs available for browser automation. Running default browser automation..."
            )
            asyncio.run(browser_main(llm))
        print("✅ Browser Automation Completed.")
    else:
        print("⏭️  Skipping Browser Automation.")

    print("\n🎉 JobAgent execution completed!")


if __name__ == "__main__":
    main(parse_args())
