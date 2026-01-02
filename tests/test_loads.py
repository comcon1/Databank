"""
`test_loads` tests ONLY functions related to downloading files and/or resolving links.

NOTE: globally import of fairmd-lipids is **STRICTLY FORBIDDEN** because it
      breaks the substitution of global path folders
"""

import os
import requests
import responses
import stat
import sys

import pytest
import pytest_check as check

# run only without mocking data
pytestmark = [pytest.mark.nodata, pytest.mark.min]


class TestDownloadWithProgressWithRetry:
    url = "https://example.org/file.bin"
    fname = "file.bin"

    @responses.activate
    def test_download_success(self, tmp_path):
        import fairmd.lipids.databankio as dio

        body = b"abcdef"
        dest = os.path.join(str(tmp_path), self.fname)

        responses.add(
            responses.GET,
            self.url,
            status=200,
            body=body,
            headers={"Content-Length": str(len(body))},
        )

        dio.download_with_progress_with_retry(self.url, dest)

        check.is_true(os.path.isfile(dest), "Success download must create a file")
        check.equal(open(dest, "br").read(), body, "Success download must get predefined content")

    @pytest.mark.parametrize(
        "fsize, dsize",
        [
            (1500, 1000),
            (700, 700),
            (1000, 1000),
        ],
    )
    @responses.activate
    def test_download_dry_run(self, tmp_path, fsize, dsize):
        import fairmd.lipids.databankio as dio

        dest = os.path.join(str(tmp_path), self.fname)
        body = b"x" * fsize

        responses.add(
            responses.GET,
            self.url,
            status=200,
            body=body,
            headers={"Content-Length": str(len(body))},
        )

        status = dio.download_with_progress_with_retry(
            self.url,
            dest,
            stop_after=1000,
        )

        check.is_true(os.path.isfile(dest), "Dry-run mode must create a file")
        check.equal(os.stat(dest).st_size, dsize, "Stop-after mode must download not more than some number of bytes")


class TestDownloadResourceFromUri:
    url = "https://example.org/file.bin"
    fname = "file.bin"

    @responses.activate
    def test_download_success(self, tmp_path):
        import fairmd.lipids.databankio as dio

        body = b"abcdef"
        dest = os.path.join(str(tmp_path), self.fname)

        responses.add(
            responses.GET,
            self.url,
            status=200,
            body=body,
            headers={"Content-Length": str(len(body))},
        )

        status = dio.download_resource_from_uri(self.url, dest)

        check.equal(status, 0, "Success download must return zero")
        check.is_true(os.path.isfile(dest), "Success download must create a file")
        check.equal(open(dest, "br").read(), body, "Success download must get predefined content")

    @pytest.mark.parametrize(
        "fsize, dsize",
        [
            (52400, 0),
            (-52400, -52400),
        ],
    )
    @responses.activate
    def test_download_dry_run(self, tmp_path, fsize, dsize):
        import fairmd.lipids.databankio as dio

        dest = os.path.join(str(tmp_path), self.fname)
        body = b"x" * (dio.MAX_DRYRUN_SIZE + fsize)

        responses.add(
            responses.GET,
            self.url,
            status=200,
            body=body,
            headers={"Content-Length": str(len(body))},
        )

        status = dio.download_resource_from_uri(
            self.url,
            dest,
            dry_run_mode=True,
        )

        check.equal(status, 0, "Dry-run mode must work")
        check.is_true(os.path.isfile(dest), "Dry-run mode must create a file")
        check.equal(
            os.stat(dest).st_size,
            dio.MAX_DRYRUN_SIZE + dsize,
            "Dry-run mode must download not more than some number of bytes",
        )


class TestGetFileSize:
    url = "https://example.org/file.bin"

    @responses.activate
    def test_no_content_length(self):
        import fairmd.lipids.databankio as dio

        responses.add(
            responses.GET,
            self.url,
            status=200,
            headers={},  # no Content-Length
        )

        size = dio._get_file_size_with_retry(self.url)

        assert size == 0

    @responses.activate
    def test_ok(self):
        import fairmd.lipids.databankio as dio

        responses.add(
            responses.GET,
            self.url,
            status=200,
            headers={"Content-Length": "1234"},
        )

        size = dio._get_file_size_with_retry(self.url)

        assert size == 1234
        assert len(responses.calls) == 1


class TestResolveZenodoFileUrl:
    def test_badDOI(self):
        import fairmd.lipids.databankio as dio

        # test if bad DOI fails
        with pytest.raises(requests.exceptions.HTTPError, match="404") as _:
            dio.resolve_zenodo_file_url("10.5281/zenodo.8435a", "a", validate_uri=True)
        # bad DOI doesn't fail if not to check
        assert (
            dio.resolve_zenodo_file_url("10.5281/zenodo.8435a", "a.txt", validate_uri=False)
            == "https://zenodo.org/record/8435a/files/a.txt"
        )

    def test_goodDOI(self):
        import fairmd.lipids.databankio as dio

        # good DOI works properly
        assert (
            dio.resolve_zenodo_file_url("10.5281/zenodo.8435138", "pope-md313rfz.tpr", validate_uri=True)
            == "https://zenodo.org/record/8435138/files/pope-md313rfz.tpr"
        )

    @pytest.mark.parametrize(
        "name, statuses, expected_exception",
        [
            ("transient 503 succeeds", [503, 503, 200], None),
            ("persistent 503 fails", [503] * 200, requests.exceptions.RetryError),
            ("403 fails immediately", [403], requests.exceptions.HTTPError),
        ],
    )
    @responses.activate
    def test_retry_logic(self, name, statuses, expected_exception):
        import fairmd.lipids.databankio as dio

        print(f"Testing resolve_doi_url with {name}", file=sys.stderr)
        url = "https://zenodo.org/record/8435138/files/a.txt"

        for status in statuses:
            responses.add(responses.GET, url, status=status)

        if expected_exception:
            with pytest.raises(expected_exception):
                dio.resolve_zenodo_file_url("10.5281/zenodo.8435138", "a.txt", validate_uri=True)
        else:
            dio.resolve_zenodo_file_url("10.5281/zenodo.8435138", "a.txt", validate_uri=True)

            assert len(responses.calls) == min(10, len(statuses))


# TODO file sha1 hash
