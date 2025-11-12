"""Crawling webpages with LLM-based structured data extraction."""

import os
import shutil
import asyncio
import argparse
import datetime
import subprocess
import getpass
import psutil

from langchain_ollama import ChatOllama
import torch
from tqdm import tqdm
from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CrawlResult,
    BrowserConfig,
)
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

from utils.colors import TColors
from utils.logging import log_extraction
from utils.structures import Extraction

WEBSITE_LIST: list[str] = [
    "https://docs.langchain.com/",
]
LOGGING_PATH: str = "./logs/"


async def main(
    device: str = "cpu",
    model: str = "llama3.1",
    crawl_depth: int = 1,
):
    """
    Main function for crawling and extracting structured data using LLMs.
    
    """
    # ──────────────────────────── set devices and print informations ─────────────────────────
    # set the devices correctly
    if "cpu" in device:
        device = torch.device("cpu", 0)
    elif "cuda" in device and torch.cuda.is_available():
        if "cuda" not in device.split(":")[-1]:
            device = torch.device("cuda", int(device.split(":")[-1]))
        else:
            device = torch.device("cuda", 0)
    elif "mps" in device and torch.backends.mps.is_available():
        if "mps" not in device.split(":")[-1]:
            device = torch.device("mps", int(device.split(":")[-1]))
        else:
            device = torch.device("mps", 0)
    else:
        print(
            f"{TColors.WARNING}Warning{TColors.ENDC}: Device {TColors.OKCYAN}{device} "
            f"{TColors.ENDC}is not available. Setting device to CPU instead."
        )
        device = torch.device("cpu", 0)

    # activate crawl4ai
    subprocess.call("crawl4ai-setup", shell=True)

    # pull the ollama model
    subprocess.call(f"ollama pull {model}", shell=True)

    # have a nice system status print
    print(
        "\n"
        + f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}System Information"
        + f"{TColors.ENDC} "
        + "#" * (shutil.get_terminal_size().columns - 23)
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Date{TColors.ENDC}: "
        + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}System{TColors.ENDC}: "
        f"{torch.get_num_threads()} CPU cores with {os.cpu_count()} threads and "
        f"{torch.cuda.device_count()} GPUs on user: {getpass.getuser()}"
    )
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Device{TColors.ENDC}: {device}")
    if (device == "cuda" or torch.device("cuda", 0)) and torch.cuda.is_available():
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}GPU Memory{TColors.ENDC}: "
            f"{torch.cuda.mem_get_info()[1] // 1024**2} MB"
        )
    elif (
        device == "mps" or torch.device("mps", 0)
    ) and torch.backends.mps.is_available():
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Shared Memory{TColors.ENDC}: "
            f"{psutil.virtual_memory()[0] // 1024**2} MB"
        )
    else:
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}CPU Memory{TColors.ENDC}: "
            f"{psutil.virtual_memory()[0] // 1024**2} MB"
        )
    print(
        f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Parameters"
        + f"{TColors.ENDC} "
        + "#" * (shutil.get_terminal_size().columns - 14)
    )
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Model{TColors.ENDC}: {model}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Crawl Depth{TColors.ENDC}: {crawl_depth}")
    print("#" * shutil.get_terminal_size().columns + "\n")


    # ------------------------------ Crawling -------------------------------
    # set up the crawler configurations and crawl the websites
    for url in WEBSITE_LIST:
        print(f"{TColors.HEADER}{TColors.BOLD}Starting Website Crawl{TColors.ENDC}")
        browser_config = BrowserConfig(
            headless=True
        )
        crawler_config = CrawlerRunConfig(
            deep_crawl_strategy=BFSDeepCrawlStrategy(max_depth=crawl_depth, include_external=False),
            scraping_strategy=LXMLWebScrapingStrategy(),
            verbose=True,
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            crawling_results: CrawlResult = await crawler.arun(
                url=url, config=crawler_config
            )

        print(f"{TColors.OKGREEN}Crawling completed successfully!{TColors.ENDC}\n")
        print(f"{TColors.OKBLUE}Total Pages Crawled: {len(crawling_results)}{TColors.ENDC}\n")

        # -------------------------- LLM-based Extraction -------------------------
        # use the LLM to check for PII information in every crawled page
        print(f"{TColors.HEADER}{TColors.BOLD}Starting LLM-based Extraction{TColors.ENDC}")

        extraction_list: list[Extraction] = []

        for page in tqdm(crawling_results, desc="Extracting Information from Pages"):
            llm = ChatOllama(
                model=model,
                temperature=0,
                device=device,
            )
            llm = llm.with_structured_output(Extraction)

            webpage_content = page.markdown.raw_markdown
            if webpage_content is None:
                print(f"{TColors.FAIL}{page.url} has no content!{TColors.ENDC}")
                continue

            messages = [
                (
                    "system",
                    """
                    You are an expert web data extractor.
                    You will help the user to search for PII handling information on websites.
                    """,
                ),
                (
                    "human",
                    """
                    Analyze the following webpage content and determine if it contains any 
                    information related to the handling of Personally Identifiable Information 
                    (PII). Remember, Personally Identifiable Information (PII) is any information 
                    that directly or indirectly identifies, relates to, describes, or is capable of 
                    being associated with a particular individual. Do not look for PII data itself, 
                    but for information about how PII data is handled, processed, stored, or 
                    protected. The webpage (converted to markdown) looks as follows:

                    """+webpage_content,
                ),
            ]

            response = llm.invoke(messages)

            if response is None:
                print(f"{TColors.FAIL}LLM failed processing {page.url}{TColors.ENDC}")
                continue

            response.url = page.url
            extraction_list.append(response)

        # log the extraction result
        log_extraction(
            extraction_list=extraction_list,
            output_name=f"{url}_extractions.json",
            log_path=LOGGING_PATH,
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PII Crawl with LLM-based Extraction")
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda",
        help="specifies the device to run the computations on (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="llama3.1",
        help="specifies the model to use for inference",
    )
    parser.add_argument(
        "--crawl_depth",
        "-cd",
        type=int,
        default=15,
        help="specifies the maximum crawl depth for deep crawling",
    )
    args = parser.parse_args()
    asyncio.run(
        main(**vars(args))
    )
