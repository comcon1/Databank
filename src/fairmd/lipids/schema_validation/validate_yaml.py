import argparse
import json
import logging
import os
import sys
from typing import Literal

import yaml
from jsonschema import Draft202012Validator, FormatChecker

"""
Tools for validating YAML files against predefined JSON Schemas.

When run as a script, this module validates one or more YAML files:
    python -m fairmd.lipids.schema_validation.validate_yaml --schema readme README.yaml
"""


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

default_info_schema_path = os.path.join(os.path.dirname(__file__), "schema", "info_yml_schema.json")
default_readme_yaml_schema_path = os.path.join(os.path.dirname(__file__), "schema", "readme_yaml_schema.json")


SchemaType = Literal["info", "readme"]


def validate_info_dict(instance: dict, schema_path: str = default_info_schema_path):
    """
    Validate an info dict against a schema dict.
    Returns a list of jsonschema.ValidationError objects which is empty with valid.
    """
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    return list(validator.iter_errors(instance))


def validate_info_file(info_file_path: str, schema_path: str = default_info_schema_path):
    """
    Validate an info file (YML/YAML) on disk against a JSON schema file.
    Returns a list of ValidationError objects (empty if valid).
    """
    with open(info_file_path, encoding="utf-8") as f:
        instance = yaml.safe_load(f)
    return validate_info_dict(instance, schema_path)


def validate_readme_dict(instance: dict, schema_path: str = default_readme_yaml_schema_path):
    """
    Validate a README.yaml dict against the README JSON schema.

    Returns a list of jsonschema.ValidationError objects (empty if valid).
    """
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    return list(validator.iter_errors(instance))


def validate_readme_yaml_file(readme_file_path: str, schema_path: str = default_readme_yaml_schema_path):
    """
    Validate a README.yaml file on disk against the README JSON schema.

    Returns a list of ValidationError objects (empty if valid).
    """
    with open(readme_file_path, encoding="utf-8") as f:
        instance = yaml.safe_load(f)
    return validate_readme_dict(instance, schema_path)


def run_file(path: str, schema_type: SchemaType) -> int:
    """
    Runs schema validation on an info.yml or README.yaml file.
    Will return error codes without throwing exceptions to be able to report on multiple files without stopping.

    :param path: path of file to validate
    :type path: str
    :param schema_type: info or readme
    :type schema_type: str
    :return: error code
    :rtype: int

    Exit codes:
      0 = valid
      1 = schema invalid
      2 = file missing / not a file
      3 = could not read / parse / other runtime error
    """
    if not os.path.isfile(path):
        logger.error("File not found (or not a file): %s", path)
        return 2

    try:
        if schema_type == "info":
            errors = validate_info_file(path)
        else:
            errors = validate_readme_yaml_file(path)
    except (OSError, yaml.YAMLError, json.JSONDecodeError) as e:
        logger.error("ERROR: %s", path)
        logger.error("  -> %s: %s", type(e).__name__, e)
        return 3

    if not errors:
        logger.info(f"OK: {path}")
        return 0

    logger.info(f"INVALID: {path}")
    for err in errors:
        keys = ".".join(str(p) for p in err.path) if err.path else "<root>"
        logger.info(f"  -> {err.message} (at {keys})")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate YAML files against FAIRMD info / README schemas.")
    parser.add_argument(
        "--schema",
        "-s",
        choices=["info", "readme"],
        default="readme",
        help="Schema to use: 'readme' (default) or 'info'.",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="YAML files to validate.",
    )

    args = parser.parse_args()

    exit_code = 0
    for f in args.files:
        path = os.path.normpath(f)
        code = run_file(path, args.schema)
        if code > exit_code:
            exit_code = code

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
