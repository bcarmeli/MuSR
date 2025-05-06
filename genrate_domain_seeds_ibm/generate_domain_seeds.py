"""
PromptEase + RITSLiteLLMClient decorator example with metadata-to-file hook.

Demonstrates:
  - Sync single calls
  - Sync batch calls
  - Async single calls
  - Async batch calls

This time we drive completions via `litellm` under the hood.
"""

import os
import time
import json
import asyncio
import logging
from dotenv import load_dotenv
# import weave

from promptease import llm
from promptease.clients.rits.rits_client import RITSLiteLLMClient

# ─── Setup logging and load .env ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
api_key = os.getenv("RITS_API_KEY")
if not api_key:
    raise EnvironmentError("Please set RITS_API_KEY in examples/.env")

# ─── Initialize RITSLiteLLMClient ───────────────────────────────────────────────
MODEL_NAME = "ibm-granite/granite-3.1-8b-instruct"
URL_MODEL = "granite-3-1-8b-instruct"
FULL_URL = f"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{URL_MODEL}/v1"

# Initialize the Weave client
# client = weave.init("PromptEase")

# Build the client. We pass our HF tokenizer to get nice chat‐style formatting.
rits_client = RITSLiteLLMClient(
    model_name=MODEL_NAME,
    # weave_client=client,
    api_base=FULL_URL,
    api_key=api_key,
    headers={"RITS_API_KEY": api_key},
    max_tokens=1000,
    guided_decoding_backend="xgrammar",
)

# If you want to change parameters, you can also do that here.
rits_client.set_parameters(
    temperature=0.6,
    min_tokens=1,
)

# ─── JSON Schema for generating fact items ──────────────────────────────────────
SEED_FACT_SCHEMA = {
    "title": "List of items",
    "description": "List of items",
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["items"]
}

# ─── Hook: save each metadata dict to a JSON file ───────────────────────────────
METADATA_DIR = os.path.join(os.path.dirname(__file__), "rits_metadata")
os.makedirs(METADATA_DIR, exist_ok=True)


def save_metadata_to_file(meta: dict):
    """
    Hook function: writes the metadata dictionary to a timestamped JSON file.
    """
    ts = int(time.time() * 1000)
    fn = meta.get("fn", "unknown")
    mode = meta.get("mode", "unknown")
    idx = meta.get("batch_index", 0)
    fname = f"{fn}_{mode}_{idx}_{ts}.json"
    path = os.path.join(METADATA_DIR, fname)
    try:
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.debug(f"Saved metadata to {path}")
    except Exception as e:
        logger.error(f"Failed to save metadata to file: {e}")


# ─── 1) Sync single call with @llm + JSON Schema + file hook ─────────────────
@llm(
    rits_client,
    schema=SEED_FACT_SCHEMA,
    retries=5,
    cache_enabled=False,
    log_level=logging.DEBUG,
    hooks=[save_metadata_to_file],
)
def create_crime_scences() -> dict:
    """Return a list of scenes in which a crime might happen. Please return a list of at least 100 places.
    Make sure to return a singleton (e.g. office) scene name, and not plural, namely 'office' and not 'offices'.
    """

@llm(
    rits_client,
    schema=SEED_FACT_SCHEMA,
    retries=5,
    cache_enabled=False,
    log_level=logging.DEBUG,
    hooks=[save_metadata_to_file],
)
def create_female_names() -> dict:
    """ Return a list of female names. Please return at least 200 names.
    """

@llm(
    rits_client,
    schema=SEED_FACT_SCHEMA,
    retries=5,
    cache_enabled=False,
    log_level=logging.DEBUG,
    hooks=[save_metadata_to_file],
)
def create_male_names() -> dict:
    """ Return a list of male names. Please return at least 200 names.
    """

@llm(
    rits_client,
    schema=SEED_FACT_SCHEMA,
    retries=5,
    cache_enabled=False,
    log_level=logging.DEBUG,
    hooks=[save_metadata_to_file],
)
def create_female_relationships() -> dict:
    """ Return a list of female occupations and relationships.
    Here are few examples:
        "Adoptive Mother",
        "Camping Buddy",
        "Director",
        "Fashion Designer",
        "Partner In Crime",
        "Twin Sister",
    Please return at least 200 names.
    """

@llm(
    rits_client,
    schema=SEED_FACT_SCHEMA,
    retries=5,
    cache_enabled=False,
    log_level=logging.DEBUG,
    hooks=[save_metadata_to_file],
)
def create_male_relationships() -> dict:
    """ Return a list of male occupations and relationships.
    Here are few examples:
        "Grandpa",
        "Biologist",
        "Soldier",
        "Beekeeper",
        "Music teacher",
        "Chauffeur"
    Please return at least 200 names.
    """

@llm(
    rits_client,
    schema=SEED_FACT_SCHEMA,
    retries=5,
    cache_enabled=False,
    log_level=logging.DEBUG,
    hooks=[save_metadata_to_file],
)
def create_murder_weapons() -> dict:
    """
    Please create a least 50 possible murder weapons.
    """

@llm(
    rits_client,
    schema=SEED_FACT_SCHEMA,
    retries=5,
    cache_enabled=False,
    log_level=logging.DEBUG,
    hooks=[save_metadata_to_file],
)
def create_family_relationships() -> dict:
    """
    Please create a large list of family relations.
    Return only the relation name, without any description.
    Few examples are:
    "Sister", "Brother", "Aunt", "Uncle", "Cousin", "Grandmother", "Grandfather", "Niece"
    """

@llm(
    rits_client,
    schema=SEED_FACT_SCHEMA,
    retries=5,
    cache_enabled=False,
    log_level=logging.DEBUG,
    hooks=[save_metadata_to_file],
)
def create_strong_motivess() -> dict:
    """
    Please create a large list of murder motives.
    Few examples are:
    "To avoid a conviction",
    "Fear",
    "To avoid a debt",
    "To prevent a marriage",
    """
@llm(
    rits_client,
    schema=SEED_FACT_SCHEMA,
    retries=5,
    cache_enabled=False,
    log_level=logging.DEBUG,
    hooks=[save_metadata_to_file],
)
def create_suspicious_facts() -> dict:
    """
    Please create a list of suspicious behaviors.
    Few examples are:
    "Consistently sneaks out at night.",
    "Has an unusual fascination with true crime documentaries.",
    "Is estranged from their family.",
    """
# print("\n--- Sync single call ---")
# resp = get_crime_scences()
# print(resp)

# # ─── 2) Sync batch call (manual flush) ────────────────────────────────────────
# @llm(
#     rits_client,
#     schema=SEED_FACT_SCHEMA,
#     batch=True,
#     auto_flush=False,
#     retries=2,
#     log_level=logging.DEBUG,
#     hooks=[save_metadata_to_file],
# )
# def batch_capitals(country: str) -> dict:
#     """Return the country and its capital information for '{country}'."""
#
#
# # Enqueue three calls (returns Future placeholders)
# _ = [
#     batch_capitals("Germany"),
#     batch_capitals("Italy"),
#     batch_capitals("Spain"),
# ]
# # Now flush and collect
# batch_results = batch_capitals.flush()
# print("\n--- Sync batch call ---")
# for i, out in enumerate(batch_results):
#     print(f"[{i}] {out}")
#
#
# # ─── 3) Async single call ─────────────────────────────────────────────────────
# @llm(
#     rits_client,
#     schema=SEED_FACT_SCHEMA,
#     asynchronous=True,
#     retries=2,
#     cache_enabled=False,
#     log_level=logging.DEBUG,
#     hooks=[save_metadata_to_file],
# )
# async def async_capital(country: str) -> dict:
#     """Return the country and its capital information for '{country}'."""
#
#
# async def run_async_single():
#     print("\n--- Async single call ---")
#     out = await async_capital("Portugal")
#     print(out)
#
#
# # ─── 4) Async batch call (manual flush) ───────────────────────────────────────
# @llm(
#     rits_client,
#     schema=SEED_FACT_SCHEMA,
#     batch=True,
#     auto_flush=False,
#     asynchronous=True,
#     retries=2,
#     log_level=logging.DEBUG,
#     hooks=[save_metadata_to_file],
# )
# async def batch_async_capitals(country: str) -> dict:
#     """Return the country and its capital information for '{country}'."""
#
#
# async def run_async_batch():
#     # Enqueue three calls (must await each to register the future)
#     _ = [
#         await batch_async_capitals("Norway"),
#         await batch_async_capitals("Sweden"),
#         await batch_async_capitals("Denmark"),
#     ]
#     # Flush them together
#     results = await batch_async_capitals.flush_async()
#     print("\n--- Async batch call ---")
#     for i, r in enumerate(results):
#         print(f"[{i}] {r}")
#

async def main():
    for _ in range(4):
        try:
            create_crime_scences()
            create_female_names()
            create_female_relationships()
            create_male_names()
            create_male_relationships()
            create_murder_weapons()
            create_family_relationships()
            create_strong_motivess()
            create_suspicious_facts()
        except Exception as e:
            print(e)
        # await run_async_single()
        # await run_async_batch()



if __name__ == "__main__":
    asyncio.run(main())
