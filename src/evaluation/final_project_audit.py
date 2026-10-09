from pathlib import Path
import ast
import csv
import re
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"

AUDIT_OUTPUT_PATH = (
    OUTPUTS_DIR
    / "tables"
    / "final_reproducibility_audit.csv"
)

DECISION_DUPLICATES_PATH = (
    OUTPUTS_DIR
    / "tables"
    / "final_decision_log_duplicates.csv"
)

METHODOLOGY_DUPLICATES_PATH = (
    OUTPUTS_DIR
    / "tables"
    / "final_methodology_heading_duplicates.csv"
)

IMPORT_AUDIT_PATH = (
    OUTPUTS_DIR
    / "tables"
    / "final_dependency_audit.csv"
)


EXPECTED_FILES = [
    "requirements.txt",

    "docs/methodology_log.md",
    "docs/decision_log.md",

    "src/evaluation/metrics.py",
    "src/evaluation/build_final_synthesis.py",
    "src/evaluation/final_project_audit.py",

    "src/data/analyze_eia930_context.py",

    "src/models/train_comstock_transfer_model.py",
    "src/models/train_comstock_transfer_model_light.py",
    "src/models/evaluate_comstock_to_bdg2_transfer.py",

    "outputs/tables/nrel_final_test_metrics.csv",
    "outputs/tables/comstock_final_test_metrics.csv",
    "outputs/tables/bdg2_final_test_metrics.csv",

    "outputs/tables/comstock_shared_feature_validation_metrics_100trees.csv",
    "outputs/tables/comstock_to_bdg2_transfer_metrics.csv",
    "outputs/tables/comstock_to_bdg2_transfer_metrics_by_building.csv",
    "outputs/tables/comstock_to_bdg2_transfer_win_summary.csv",

    "outputs/tables/eia930_california_summary.csv",
    "outputs/tables/eia930_california_annual_summary.csv",
    "outputs/tables/eia930_california_monthly_profile.csv",
    "outputs/tables/eia930_california_hourly_profile.csv",
    "outputs/tables/eia930_california_peak_demand_hours.csv",
    "outputs/tables/eia930_california_generation_mix.csv",
    "outputs/tables/eia930_california_forecast_error_summary.csv",

    "outputs/tables/final_model_performance_summary.csv",
    "outputs/tables/final_mae_improvement_vs_seasonal.csv",
    "outputs/tables/final_transfer_comparison.csv",
    "outputs/tables/final_project_key_findings.csv",

    "outputs/figures/eia930_average_demand_by_hour.png",
    "outputs/figures/eia930_average_demand_by_month.png",
    "outputs/figures/eia930_generation_mix.png",
    "outputs/figures/final_mae_improvement_vs_seasonal.png",
    "outputs/figures/final_simulated_vs_real_transfer.png",

    "outputs/models/comstock_shared_feature_random_forest_100trees.joblib",
]


REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "joblib": "joblib",
    "matplotlib": "matplotlib",
    "pyarrow": "pyarrow",
    "openpyxl": "openpyxl",
    "boto3": "boto3",
}


PLACEHOLDER_PATTERNS = [
    r"\bTODO\b",
    r"\bFIXME\b",
    r"\bYOUR_[A-Z0-9_]+\b",
    r"\bREPLACE_ME\b",
    r"\bCHANGEME\b",
]


def add_result(
    results: list[dict[str, object]],
    category: str,
    item: str,
    status: str,
    detail: str,
) -> None:

    results.append(
        {
            "category": category,
            "item": item,
            "status": status,
            "detail": detail,
        }
    )


def audit_expected_files(
    results: list[dict[str, object]],
) -> None:

    print()
    print("Checking expected project files...")

    for relative_path in EXPECTED_FILES:

        path = PROJECT_ROOT / relative_path

        if path.exists():
            add_result(
                results,
                "expected_file",
                relative_path,
                "PASS",
                "File exists.",
            )
        else:
            add_result(
                results,
                "expected_file",
                relative_path,
                "FAIL",
                "Expected file is missing.",
            )


def audit_python_syntax(
    results: list[dict[str, object]],
) -> None:

    print()
    print("Checking Python syntax...")

    python_files = sorted(
        SRC_DIR.rglob("*.py")
    )

    if not python_files:
        add_result(
            results,
            "python_syntax",
            "src",
            "FAIL",
            "No Python files found under src.",
        )
        return

    failed = 0

    for path in python_files:

        relative_path = path.relative_to(
            PROJECT_ROOT
        )

        try:
            source = path.read_text(
                encoding="utf-8"
            )

            ast.parse(
                source,
                filename=str(path),
            )

            add_result(
                results,
                "python_syntax",
                str(relative_path),
                "PASS",
                "Python syntax parsed successfully.",
            )

        except Exception as exc:

            failed += 1

            add_result(
                results,
                "python_syntax",
                str(relative_path),
                "FAIL",
                str(exc),
            )

    print(
        f"Python files checked: {len(python_files):,}"
    )

    print(
        f"Syntax failures: {failed:,}"
    )


def load_requirements() -> set[str]:

    path = PROJECT_ROOT / "requirements.txt"

    if not path.exists():
        return set()

    packages: set[str] = set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for raw_line in handle:

            line = raw_line.strip()

            if (
                not line
                or line.startswith("#")
                or line.startswith("-")
            ):
                continue

            package = re.split(
                r"[<>=!~\[\]; ]",
                line,
                maxsplit=1,
            )[0]

            if package:
                packages.add(
                    package.lower()
                    .replace("_", "-")
                )

    return packages


def collect_imports() -> set[str]:

    imports: set[str] = set()

    for path in SRC_DIR.rglob("*.py"):

        try:
            source = path.read_text(
                encoding="utf-8"
            )

            tree = ast.parse(
                source,
                filename=str(path),
            )

        except Exception:
            continue

        for node in ast.walk(
            tree
        ):

            if isinstance(
                node,
                ast.Import,
            ):

                for alias in node.names:

                    imports.add(
                        alias.name.split(".")[0]
                    )

            elif isinstance(
                node,
                ast.ImportFrom,
            ):

                if node.module:

                    imports.add(
                        node.module.split(".")[0]
                    )

    return imports


def audit_dependencies(
    results: list[dict[str, object]],
) -> None:

    print()
    print("Checking requirements.txt...")

    requirements = load_requirements()
    imports = collect_imports()

    rows: list[
        dict[str, object]
    ] = []

    for import_name, package_name in (
        REQUIRED_PACKAGES.items()
    ):

        normalized_package = (
            package_name
            .lower()
            .replace("_", "-")
        )

        imported = (
            import_name in imports
        )

        listed = (
            normalized_package
            in requirements
        )

        if imported and listed:
            status = "PASS"
            detail = (
                "Imported by project and listed "
                "in requirements.txt."
            )

        elif imported and not listed:
            status = "FAIL"
            detail = (
                "Imported by project but missing "
                "from requirements.txt."
            )

        elif not imported and listed:
            status = "PASS"
            detail = (
                "Listed in requirements.txt; "
                "not detected in src imports."
            )

        else:
            status = "WARN"
            detail = (
                "Not detected in src imports and "
                "not listed in requirements.txt."
            )

        rows.append(
            {
                "import_name":
                    import_name,
                "package_name":
                    package_name,
                "imported_in_src":
                    imported,
                "listed_in_requirements":
                    listed,
                "status":
                    status,
            }
        )

        add_result(
            results,
            "dependency",
            package_name,
            status,
            detail,
        )

    dependency_df = pd.DataFrame(
        rows
    )

    dependency_df.to_csv(
        IMPORT_AUDIT_PATH,
        index=False,
    )


def audit_placeholders(
    results: list[dict[str, object]],
) -> None:

    print()
    print(
        "Checking source files for placeholders..."
    )

    files_to_scan = list(
        SRC_DIR.rglob("*.py")
    )

    for optional_path in [
        PROJECT_ROOT / "README.md",
    ]:

        if optional_path.exists():
            files_to_scan.append(
                optional_path
            )

    matches_found = 0

    for path in files_to_scan:

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

        except Exception as exc:

            add_result(
                results,
                "placeholder_scan",
                str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "WARN",
                f"Could not read file: {exc}",
            )

            continue

        file_matches: list[str] = []

        for pattern in PLACEHOLDER_PATTERNS:

            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):

                file_matches.append(
                    pattern
                )

        if file_matches:

            matches_found += 1

            add_result(
                results,
                "placeholder_scan",
                str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "WARN",
                (
                    "Potential placeholder patterns: "
                    + ", ".join(
                        file_matches
                    )
                ),
            )

    if matches_found == 0:

        add_result(
            results,
            "placeholder_scan",
            "src and README",
            "PASS",
            "No common placeholder strings found.",
        )


def audit_hardcoded_windows_paths(
    results: list[dict[str, object]],
) -> None:

    print()
    print(
        "Checking source files for hard-coded user paths..."
    )

    matches: list[str] = []

    pattern = re.compile(
        r"[A-Za-z]:\\Users\\",
        re.IGNORECASE,
    )

    for path in SRC_DIR.rglob("*.py"):

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        if pattern.search(
            text
        ):

            matches.append(
                str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                )
            )

    if matches:

        for relative_path in matches:

            add_result(
                results,
                "hardcoded_path",
                relative_path,
                "WARN",
                "Possible hard-coded Windows user path found.",
            )

    else:

        add_result(
            results,
            "hardcoded_path",
            "src",
            "PASS",
            "No C:\\Users\\ hard-coded paths detected.",
        )


def audit_decision_log(
    results: list[dict[str, object]],
) -> None:

    print()
    print(
        "Checking decision log for duplicate IDs..."
    )

    path = DOCS_DIR / "decision_log.md"

    if not path.exists():

        add_result(
            results,
            "decision_log",
            "decision_log.md",
            "FAIL",
            "Decision log not found.",
        )

        return

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    ids = re.findall(
        r"^\|\s*(D\d+)\s*\|",
        text,
        flags=re.MULTILINE,
    )

    counts = pd.Series(
        ids,
        dtype="object",
    ).value_counts()

    duplicates = (
        counts.loc[
            counts > 1
        ]
        .rename_axis(
            "decision_id"
        )
        .reset_index(
            name="occurrences"
        )
    )

    duplicates.to_csv(
        DECISION_DUPLICATES_PATH,
        index=False,
    )

    if duplicates.empty:

        add_result(
            results,
            "decision_log",
            "duplicate IDs",
            "PASS",
            "No duplicate decision IDs found.",
        )

    else:

        duplicate_list = ", ".join(
            duplicates[
                "decision_id"
            ].astype(
                str
            )
        )

        add_result(
            results,
            "decision_log",
            "duplicate IDs",
            "WARN",
            (
                "Duplicate decision IDs found: "
                f"{duplicate_list}"
            ),
        )


def audit_methodology_log(
    results: list[dict[str, object]],
) -> None:

    print()
    print(
        "Checking methodology log for duplicate section headings..."
    )

    path = DOCS_DIR / "methodology_log.md"

    if not path.exists():

        add_result(
            results,
            "methodology_log",
            "methodology_log.md",
            "FAIL",
            "Methodology log not found.",
        )

        return

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    headings = re.findall(
        r"^##\s+(.+?)\s*$",
        text,
        flags=re.MULTILINE,
    )

    normalized = [
        heading.strip()
        for heading in headings
    ]

    counts = pd.Series(
        normalized,
        dtype="object",
    ).value_counts()

    duplicates = (
        counts.loc[
            counts > 1
        ]
        .rename_axis(
            "heading"
        )
        .reset_index(
            name="occurrences"
        )
    )

    duplicates.to_csv(
        METHODOLOGY_DUPLICATES_PATH,
        index=False,
    )

    if duplicates.empty:

        add_result(
            results,
            "methodology_log",
            "duplicate headings",
            "PASS",
            "No duplicate H2 headings found.",
        )

    else:

        add_result(
            results,
            "methodology_log",
            "duplicate headings",
            "WARN",
            (
                f"{len(duplicates):,} duplicated "
                "H2 heading(s) found."
            ),
        )


def audit_final_tables(
    results: list[dict[str, object]],
) -> None:

    print()
    print(
        "Checking final synthesis tables..."
    )

    final_tables = [
        (
            "final_model_performance_summary.csv",
            [
                "source_dataset",
                "evaluation_split",
                "model",
                "MAE",
                "CVRMSE_percent",
                "MAPE_percent",
            ],
        ),
        (
            "final_mae_improvement_vs_seasonal.csv",
            [
                "dataset",
                "evaluation_split",
                "selected_model",
                "model_MAE",
                "seasonal_MAE",
                "MAE_improvement_percent",
            ],
        ),
        (
            "final_transfer_comparison.csv",
            [
                "evaluation_split",
                "model",
                "MAE",
                "CVRMSE_percent",
                "MAPE_percent",
            ],
        ),
        (
            "final_project_key_findings.csv",
            [
                "topic",
                "finding",
            ],
        ),
    ]

    for filename, required_columns in final_tables:

        path = (
            OUTPUTS_DIR
            / "tables"
            / filename
        )

        if not path.exists():
            continue

        try:

            data = pd.read_csv(
                path
            )

        except Exception as exc:

            add_result(
                results,
                "final_table",
                filename,
                "FAIL",
                f"Could not read CSV: {exc}",
            )

            continue

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:

            add_result(
                results,
                "final_table",
                filename,
                "FAIL",
                (
                    "Missing required columns: "
                    f"{missing_columns}"
                ),
            )

        elif data.empty:

            add_result(
                results,
                "final_table",
                filename,
                "FAIL",
                "Table exists but contains no rows.",
            )

        else:

            add_result(
                results,
                "final_table",
                filename,
                "PASS",
                (
                    f"{len(data):,} rows; "
                    "required columns present."
                ),
            )


def audit_key_findings_format(
    results: list[dict[str, object]],
) -> None:

    path = (
        OUTPUTS_DIR
        / "tables"
        / "final_project_key_findings.csv"
    )

    if not path.exists():
        return

    data = pd.read_csv(
        path
    )

    if "finding" not in data.columns:
        return

    combined = " ".join(
        data[
            "finding"
        ]
        .fillna("")
        .astype(
            str
        )
        .tolist()
    )

    suspicious_patterns = [
        r"\bby\d",
        r"\d%relative\b",
        r"\d%compared\b",
    ]

    found: list[str] = []

    for pattern in suspicious_patterns:

        if re.search(
            pattern,
            combined,
            flags=re.IGNORECASE,
        ):

            found.append(
                pattern
            )

    if found:

        add_result(
            results,
            "formatting",
            "final_project_key_findings.csv",
            "WARN",
            (
                "Possible missing-space formatting issue "
                "detected in generated findings."
            ),
        )

    else:

        add_result(
            results,
            "formatting",
            "final_project_key_findings.csv",
            "PASS",
            "No obvious spacing issues detected.",
        )


def audit_git_status(
    results: list[dict[str, object]],
) -> None:

    print()
    print(
        "Checking Git status..."
    )

    try:

        process = subprocess.run(
            [
                "git",
                "status",
                "--short",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    except FileNotFoundError:

        add_result(
            results,
            "git",
            "git executable",
            "WARN",
            "Git executable not found.",
        )

        return

    if process.returncode != 0:

        add_result(
            results,
            "git",
            "repository status",
            "WARN",
            process.stderr.strip(),
        )

        return

    lines = [
        line
        for line in process.stdout.splitlines()
        if line.strip()
    ]

    if lines:

        add_result(
            results,
            "git",
            "working tree",
            "WARN",
            (
                f"{len(lines):,} uncommitted "
                "file change(s) detected."
            ),
        )

        print()

        print(
            "Uncommitted Git changes:"
        )

        for line in lines:

            print(
                line
            )

    else:

        add_result(
            results,
            "git",
            "working tree",
            "PASS",
            "Git working tree is clean.",
        )


def print_summary(
    audit_df: pd.DataFrame,
) -> None:

    print()
    print(
        "=" * 72
    )

    print(
        "FINAL PROJECT AUDIT SUMMARY"
    )

    print(
        "=" * 72
    )

    counts = (
        audit_df[
            "status"
        ]
        .value_counts()
    )

    for status in [
        "PASS",
        "WARN",
        "FAIL",
    ]:

        print(
            f"{status}: "
            f"{int(counts.get(status, 0)):,}"
        )

    issues = audit_df.loc[
        audit_df[
            "status"
        ].isin(
            [
                "WARN",
                "FAIL",
            ]
        )
    ]

    if issues.empty:

        print()
        print(
            "No warnings or failures detected."
        )

    else:

        print()
        print(
            "Items requiring review:"
        )

        print(
            issues[
                [
                    "category",
                    "item",
                    "status",
                    "detail",
                ]
            ]
            .to_string(
                index=False
            )
        )


def main() -> None:

    (
        OUTPUTS_DIR
        / "tables"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    results: list[
        dict[str, object]
    ] = []

    print(
        "Running final project reproducibility audit..."
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    audit_expected_files(
        results
    )

    audit_python_syntax(
        results
    )

    audit_dependencies(
        results
    )

    audit_placeholders(
        results
    )

    audit_hardcoded_windows_paths(
        results
    )

    audit_decision_log(
        results
    )

    audit_methodology_log(
        results
    )

    audit_final_tables(
        results
    )

    audit_key_findings_format(
        results
    )

    audit_git_status(
        results
    )

    audit_df = pd.DataFrame(
        results
    )

    audit_df.to_csv(
        AUDIT_OUTPUT_PATH,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    print_summary(
        audit_df
    )

    print()
    print(
        f"Full audit: "
        f"{AUDIT_OUTPUT_PATH}"
    )

    print(
        f"Decision-log duplicates: "
        f"{DECISION_DUPLICATES_PATH}"
    )

    print(
        f"Methodology-heading duplicates: "
        f"{METHODOLOGY_DUPLICATES_PATH}"
    )

    print(
        f"Dependency audit: "
        f"{IMPORT_AUDIT_PATH}"
    )

    fail_count = int(
        (
            audit_df[
                "status"
            ]
            == "FAIL"
        ).sum()
    )

    if fail_count > 0:

        print()
        print(
            "Audit completed with failures. "
            "Review FAIL items before finalizing the repository."
        )

        sys.exit(
            1
        )

    print()
    print(
        "Audit completed without hard failures."
    )


if __name__ == "__main__":
    main()