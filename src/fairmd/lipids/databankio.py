"""
Input/Output auxilary module **fairmd.lipids.databankio**.

Input/Output auxilary module with some small usefull functions. It includes:
- Network communication.
- Downloading files.
- Checking links.
- Resolving DOIs.
- Calculating file hashes.
"""

import hashlib
import logging
import math
import os
import urllib.error
from collections.abc import Mapping
from contextlib import contextmanager

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)
MAX_DRYRUN_SIZE = 50 * 1024 * 1024  # 50 MB, max size for dry-run download


SOFTWARE_CONFIG = {
    "gromacs": {"primary": "TPR", "secondary": "TRJ"},
    "openMM": {"primary": "TRJ", "secondary": "TRJ"},
    "NAMD": {"primary": "TRJ", "secondary": "TRJ"},
}


def requests_session_with_retry(
    retries: int = 5,
    backoff: float = 10,
) -> requests.Session:
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=None,
    )
    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# --- Decorated Helper Functions for Network Requests ---


@contextmanager
def _open_url_with_retry(uri: str, backoff: float = 10, *, stream: bool = True):
    """Open a URL with a timeout and retry logic (aprivate helper).

    :param uri: The URL to open.
    :param backoff: The backoff timeout for the request in seconds.

    :return: The response object.
    """
    with requests_session_with_retry(retries=5, backoff=backoff) as session:
        response = session.get(uri, stream=stream)
        response.raise_for_status()
        try:
            yield response
        finally:
            response.close()


def get_file_size_with_retry(uri: str) -> int:
    """Fetch the size of a file from a URI with retry logic.

    :param uri: (str) The URL of the file.

    :returns: The size of the file in bytes, or 0 if the 'Content-Length'
              header is not present (int).
    """
    with _open_url_with_retry(uri) as response:
        content_length = response.headers.get("Content-Length")
        return int(content_length) if content_length else 0


def download_with_progress_with_retry(
    uri: str, dest: str, *, tqdm_title: str = "Downloading", stop_after: int | None = None
) -> None:
    """Download a file with a progress bar and retry logic.

    Uses tqdm to display a progress bar during the download.

    Args:
        uri (str): The URL of the file to download.
        dest (str): The local destination path to save the file.
        tqdm_title (str): The title used for the progress bar description.
        stop_after (int): Download max num of bytes
    """

    class RetrieveProgressBar(tqdm):
        def update_retrieve(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            return self.update(b * bsize - self.n)

    with (
        RetrieveProgressBar(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=tqdm_title,
        ) as u,
        open(dest, "wb") as f,
        _open_url_with_retry(uri) as resp,
    ):
        total = int(resp.headers.get("Content-Length", 0))
        if stop_after is not None:
            total = min(total, stop_after)
        downloaded = 0

        for chunk in resp.iter_content(8192):
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded > total:
                break
            u.update_retrieve(b=downloaded, bsize=1, tsize=total)

    if downloaded > total:
        with open(dest, "rb+") as f:
            f.truncate(total)


# --- Main Functions ---


def download_resource_from_uri(
    uri: str,
    dest: str,
    *,
    override_if_exists: bool = False,
    dry_run_mode: bool = False,
) -> int:
    """Download file resource from a URI to a local destination.

    Checks if the file already exists and has the same size before downloading.
    Can also perform a partial "dry-run" download.

    Args:
        uri (str): The URL of the file resource.
        dest (str): The local destination path to save the file.
        override_if_exists (bool): If True, the file will be re-downloaded
            even if it already exists. Defaults to False.
        dry_run_mode (bool): If True, only a partial download is performed
            (up to MAX_DRYRUN_SIZE). Defaults to False.

    Returns
    -------
        int: A status code indicating the result.
            0: Download was successful.
            1: Download was skipped because the file already exists.
            2: File was re-downloaded due to a size mismatch.

    Raises
    ------
        ConnectionError: An error occurred after multiple download attempts.
        OSError: The downloaded file size does not match the expected size.
    """
    fi_name = uri.split("/")[-1]
    return_code = 0

    if os.path.isdir(dest):
        msg = f"Destination '{dest}' is a directory, not a file."
        raise IsADirectoryError(msg)

    # Check if dest path already exists and compare file size
    if not override_if_exists and os.path.isfile(dest):
        try:
            fi_size = get_file_size_with_retry(uri)
            if fi_size == os.path.getsize(dest):
                logger.info(f"{dest}: file already exists, skipping")
                return 1
            logger.warning(
                f"{fi_name} filesize mismatch of local file '{fi_name}', redownloading ...",
            )
            return_code = 2
        except FileNotFoundError:
            logger.exception(
                f"Failed to verify file size for {fi_name}. Proceeding with redownload.",
            )
            return_code = 2

    # Download file in dry run mode
    if dry_run_mode:
        url_size = get_file_size_with_retry(uri)
        download_with_progress_with_retry(uri, dest, tqdm_title=fi_name, stop_after=MAX_DRYRUN_SIZE)
        return 0

    # Download with progress bar and check for final size match
    url_size = get_file_size_with_retry(uri)
    download_with_progress_with_retry(uri, dest, tqdm_title=fi_name)

    size = os.path.getsize(dest)
    if url_size != 0 and url_size != size:
        msg = f"Downloaded filesize mismatch ({size}/{url_size} B)"
        raise OSError(msg)

    return return_code


def resolve_doi_url(doi: str, validate_uri: bool = True) -> str:
    """Resolve a DOI to a full URL and validates that it is reachable.

    Args:
        doi (str): The DOI identifier (e.g., "10.5281/zenodo.1234").
        validate_uri (bool): If True, checks if the resolved URL is a valid
            and reachable address. Defaults to True.

    Returns
    -------
        str: The full, validated DOI link (e.g., "https://doi.org/...").

    Raises
    ------
        urllib.error.HTTPError: If the DOI resolves to a URL, but the server
            returns an HTTP error code (e.g., 404 Not Found).
        ConnectionError: If the server cannot be reached after multiple retries.
    """
    res = "https://doi.org/" + doi

    if validate_uri:
        try:
            # The 'with' statement ensures the connection is closed
            with _open_url_with_retry(res):
                pass
        except urllib.error.HTTPError as e:
            # This specifically catches HTTP errors like 404, 500, etc.
            logger.exception(f"Validation failed for DOI {doi}. URL <{res}> returned HTTP {e.code}: {e.reason}")
            # Re-raise the specific HTTPError so the caller can handle it
            raise
        except ConnectionError:
            # This catches the final error from the retry decorator for other issues
            # (e.g., DNS failure, connection refused).
            logger.exception(f"Could not connect to <{res}> after multiple attempts.")
            raise

    return res


def resolve_download_file_url(doi: str, fi_name: str, *, validate_uri: bool = True) -> str:
    """
    Resolve a download file URI from a supported DOI and filename.

    Currently supports Zenodo DOIs.

    Args:
        doi (str): The DOI identifier for the repository (e.g.,
            "10.5281/zenodo.1234").
        fi_name (str): The name of the file within the repository.
        validate_uri (bool): If True, checks if the resolved URL is a valid
            and reachable address. Defaults to True.

    Returns
    -------
        str: The full, direct download URL for the file.

    Raises
    ------
        HTTPError or other connection errors: If the URL cannot be opened after multiple retries.
        NotImplementedError: If the DOI provider is not supported.
    """
    if "zenodo" in doi.lower():
        zenodo_entry_number = doi.split(".")[2]
        uri = f"https://zenodo.org/record/{zenodo_entry_number}/files/{fi_name}"

        if validate_uri:
            # Use the decorated helper to check if the URI exists
            # If not - it raises the exceptions
            with _open_url_with_retry(uri):
                pass
        return uri
    msg = "Repository not validated. Please upload the data for example to zenodo.org"
    raise NotImplementedError(msg)


def calc_file_sha1_hash(fi: str, step: int = 67108864, one_block: bool = True) -> str:
    """Calculate the SHA1 hash of a file.

    Reads the file in chunks to handle large files efficiently if specified.

    Args:
        fi (str): The path to the file.
        step (int): The chunk size in bytes for reading the file.
            Defaults to 64MB. Only used if `one_block` is False.
        one_block (bool): If True, reads the first `step` bytes of the file.
            If False, reads the entire file in chunks of `step` bytes.
            Defaults to True.

    Returns
    -------
        str: The hexadecimal SHA1 hash of the file content.
    """
    sha1_hash = hashlib.sha1()
    n_tot_steps = math.ceil(os.path.getsize(fi) / step)
    with open(fi, "rb") as f:
        if one_block:
            block = f.read(step)
            sha1_hash.update(block)
        else:
            with tqdm(total=n_tot_steps, desc="Calculating SHA1") as pbar:
                for byte_block in iter(lambda: f.read(step), b""):
                    sha1_hash.update(byte_block)
                    pbar.update(1)
    return sha1_hash.hexdigest()


def create_databank_directories(
    sim: Mapping,
    sim_hashes: Mapping,
    out: str,
    *,
    dry_run_mode: bool = False,
) -> str:
    """Create a nested output directory structure to save simulation results.

    The directory structure is generated based on the hashes of the simulation
    input files.

    Args:
        sim (Mapping): A dictionary containing simulation metadata, including
            the "SOFTWARE" key.
        sim_hashes (Mapping): A dictionary mapping file types (e.g., "TPR",
            "TRJ") to their hash information. The structure is expected to be
            `{'TYPE': [('filename', 'hash')]}`.
        out (str): The root output directory where the nested structure
            will be created.
        dry_run_mode (bool): If True, the directory path is resolved but
            not created. Defaults to False.

    Returns
    -------
        str: The full path to the created output directory.

    Raises
    ------
        FileExistsError: If the target output directory already exists and is
            not empty.
        NotImplementedError: If the simulation software is not supported.
        RuntimeError: If the target output directory could not be created.
    """
    # resolve output dir naming
    software: str = sim.get("SOFTWARE")

    config = SOFTWARE_CONFIG.get(software)
    if not config:
        msg = f"Sim software '{software}' not supported"
        raise NotImplementedError(msg)

    primary_hash = sim_hashes.get(config["primary"])[0][1]
    secondary_hash = sim_hashes.get(config["secondary"])[0][1]

    head_dir = primary_hash[:3]
    sub_dir1 = primary_hash[3:6]
    sub_dir2 = primary_hash
    sub_dir3 = secondary_hash

    directory_path = os.path.join(out, head_dir, sub_dir1, sub_dir2, sub_dir3)

    logger.debug(f"output_dir = {directory_path}")

    if os.path.exists(directory_path) and os.listdir(directory_path):
        msg = f"Output directory '{directory_path}' is not empty. Delete it if you wish."
        raise FileExistsError(msg)

    if not dry_run_mode:
        try:
            os.makedirs(directory_path, exist_ok=True)
        except Exception as e:
            msg = f"Could not create the output directory at {directory_path}"
            raise RuntimeError(msg) from e

    return directory_path
