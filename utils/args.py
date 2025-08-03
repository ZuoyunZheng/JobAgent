import argparse


def parse_args():
    argparser = argparse.ArgumentParser(
        description="JobAgent - Automated Job Application System"
    )
    argparser.add_argument(
        "--reload_data", action="store_true", help="Reload and re-index candidate data"
    )
    argparser.add_argument(
        "--skip_rag", action="store_true", help="Skip RAG pipeline initialization"
    )
    argparser.add_argument(
        "--skip_scraping", action="store_true", help="Skip job scraping"
    )
    argparser.add_argument(
        "--skip_browser", action="store_true", help="Skip browser automation"
    )
    argparser.add_argument(
        "--job_terms",
        default="Machine Learning",
        help="Job search terms (default: Machine Learning)",
    )
    argparser.add_argument(
        "--location", default="Germany", help="Job search location (default: Germany)"
    )
    argparser.add_argument(
        "--num_jobs",
        type=int,
        default=20,
        help="Number of jobs to scrape (default: 20)",
    )
    argparser.add_argument(
        "--save_jobs", action="store_true", help="Save scraped jobs to jobs.pkl file"
    )
    return argparser.parse_args()
