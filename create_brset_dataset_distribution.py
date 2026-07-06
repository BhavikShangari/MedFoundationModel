import pandas as pd


SOURCE_CSV = "labels_brset.csv"
OUTPUT_CSV = "brset_dataset_distribution.csv"
OUTPUT_STATS_MD = "brset_dataset_statistics.md"
RANDOM_STATE = 42
TEST_IMAGES_PER_CATEGORY = 50
IMAGES_PER_SELECTED_PATIENT = 2
TEST_CATEGORIES = [
    "no_disease",
    "single_disease",
    "multiple_disease",
]

BASE_COLUMNS = [
    "image_id",
    "patient_id",
    "patient_age",
    "patient_sex",
]

DISEASE_COLUMNS = [
    "diabetic_retinopathy",
    "macular_edema",
    "scar",
    "nevus",
    "amd",
    "vascular_occlusion",
    "hypertensive_retinopathy",
    "drusens",
    "hemorrhage",
    "retinal_detachment",
    "myopic_fundus",
    "increased_cup_disc",
    "other",
]


def disease_category(disease_count: int) -> str:
    if disease_count == 0:
        return "no_disease"
    if disease_count == 1:
        return "single_disease"
    return "multiple_disease"


def eligible_two_image_patients(df: pd.DataFrame, category: str) -> pd.Index:
    patient_groups = df.groupby("patient_id")
    group_sizes = patient_groups.size()
    category_consistent = patient_groups["disease_category"].agg(
        lambda values: values.eq(category).all()
    )
    return group_sizes[group_sizes.eq(2) & category_consistent].index


def value_count_table(series: pd.Series, total: int | None = None) -> pd.DataFrame:
    counts = series.value_counts(dropna=False).sort_index()
    if total is None:
        total = int(counts.sum())
    return pd.DataFrame(
        {
            "count": counts.astype(int),
            "percent": (counts / total * 100).round(2),
        }
    )


def markdown_table(df: pd.DataFrame, index_name: str) -> str:
    table = df.reset_index(names=index_name)
    headers = list(table.columns)
    rows = [[str(value) for value in row] for row in table.itertuples(index=False, name=None)]
    widths = [
        max(len(str(header)), *(len(row[i]) for row in rows)) if rows else len(str(header))
        for i, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(
        str(header).ljust(widths[i]) for i, header in enumerate(headers)
    ) + " |"
    separator = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def disease_prevalence_table(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows = []
    for column in DISEASE_COLUMNS:
        positive = int(df[column].sum())
        rows.append(
            {
                "disease": column,
                "positive_images": positive,
                "percent_of_images": round(positive / total * 100, 2),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("positive_images", ascending=False)
        .set_index("disease")
    )


def append_distribution(
    lines: list[str], title: str, series: pd.Series, index_name: str
) -> None:
    lines.append(f"\n## {title}\n")
    lines.append(markdown_table(value_count_table(series), index_name))


def write_statistics_report(full_df: pd.DataFrame, split_df: pd.DataFrame) -> None:
    stats_df = full_df.merge(
        split_df[["image_id", "disease_count", "disease_category", "split"]],
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    patient_overlap = (
        set(stats_df.loc[stats_df["split"].eq("test"), "patient_id"])
        & set(stats_df.loc[stats_df["split"].eq("retrieval"), "patient_id"])
    )

    lines = [
        "# BRSET Dataset Distribution Statistics",
        "",
        "Disease category definitions:",
        "",
        "- no_disease: disease_count == 0",
        "- single_disease: disease_count == 1",
        "- multiple_disease: disease_count >= 2",
        "",
        "## Overview",
        "",
        f"- Total images: {len(stats_df)}",
        f"- Unique patients: {stats_df['patient_id'].nunique()}",
        f"- Disease indicator columns: {len(DISEASE_COLUMNS)}",
        f"- Test images per category: {TEST_IMAGES_PER_CATEGORY}",
        f"- Patient overlap between retrieval and test: {len(patient_overlap)}",
    ]

    split_counts = pd.DataFrame(
        {
            "images": stats_df.groupby("split").size(),
            "patients": stats_df.groupby("split")["patient_id"].nunique(),
            "percent_images": (stats_df.groupby("split").size() / len(stats_df) * 100).round(2),
        }
    )
    lines.append("\n## Split Summary\n")
    lines.append(markdown_table(split_counts, "split"))

    category_by_split = (
        stats_df.groupby(["split", "disease_category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=TEST_CATEGORIES)
    )
    category_by_split["total"] = category_by_split.sum(axis=1)
    lines.append("\n## Disease Category By Split\n")
    lines.append(markdown_table(category_by_split, "split"))

    count_by_split = (
        stats_df.groupby(["split", "disease_count"])
        .size()
        .unstack(fill_value=0)
        .sort_index(axis=1)
    )
    count_by_split["total"] = count_by_split.sum(axis=1)
    lines.append("\n## Disease Count By Split\n")
    lines.append(markdown_table(count_by_split, "split"))

    append_distribution(
        lines,
        "Overall Disease Category Distribution",
        stats_df["disease_category"],
        "disease_category",
    )
    append_distribution(
        lines,
        "Overall Disease Count Distribution",
        stats_df["disease_count"],
        "disease_count",
    )

    lines.append("\n## Disease Label Prevalence: Entire Dataset\n")
    lines.append(markdown_table(disease_prevalence_table(stats_df), "disease"))
    for split_name in ["retrieval", "test"]:
        lines.append(f"\n## Disease Label Prevalence: {split_name.title()} Split\n")
        split_part = stats_df[stats_df["split"].eq(split_name)]
        lines.append(markdown_table(disease_prevalence_table(split_part), "disease"))

    append_distribution(lines, "Images Per Patient", stats_df.groupby("patient_id").size(), "images")

    age = stats_df["patient_age"]
    age_summary = pd.DataFrame(
        {
            "value": [
                int(age.notna().sum()),
                int(age.isna().sum()),
                round(float(age.mean()), 2),
                round(float(age.median()), 2),
                round(float(age.min()), 2),
                round(float(age.max()), 2),
            ]
        },
        index=["non_missing", "missing", "mean", "median", "min", "max"],
    )
    lines.append("\n## Patient Age Summary\n")
    lines.append(markdown_table(age_summary, "statistic"))

    for column in [
        "patient_sex",
        "exam_eye",
        "camera",
        "quality",
        "diabetes",
        "nationality",
    ]:
        if column in stats_df.columns:
            append_distribution(lines, f"{column} Distribution", stats_df[column], column)

    with open(OUTPUT_STATS_MD, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    df = pd.read_csv(SOURCE_CSV)
    split_df = df[BASE_COLUMNS + DISEASE_COLUMNS].copy()

    split_df["disease_count"] = split_df[DISEASE_COLUMNS].sum(axis=1).astype(int)
    split_df["disease_category"] = split_df["disease_count"].map(disease_category)
    split_df["split"] = "retrieval"

    if TEST_IMAGES_PER_CATEGORY % IMAGES_PER_SELECTED_PATIENT != 0:
        raise ValueError("TEST_IMAGES_PER_CATEGORY must be divisible by 2")

    patients_per_category = TEST_IMAGES_PER_CATEGORY // IMAGES_PER_SELECTED_PATIENT
    test_patient_ids = set()
    for category in TEST_CATEGORIES:
        eligible_patients = eligible_two_image_patients(split_df, category)
        if len(eligible_patients) < patients_per_category:
            raise ValueError(
                f"Need {patients_per_category} {category} patients, "
                f"found {len(eligible_patients)}"
            )
        sampled_patients = eligible_patients.to_series().sample(
            n=patients_per_category, random_state=RANDOM_STATE
        )
        test_patient_ids.update(sampled_patients)

    split_df.loc[split_df["patient_id"].isin(test_patient_ids), "split"] = "test"

    split_df.to_csv(OUTPUT_CSV, index=False)
    write_statistics_report(df, split_df)

    test_df = split_df[split_df["split"].eq("test")]
    patient_overlap = (
        set(split_df.loc[split_df["split"].eq("test"), "patient_id"])
        & set(split_df.loc[split_df["split"].eq("retrieval"), "patient_id"])
    )

    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_STATS_MD}")
    print(f"retrieval_images={int(split_df['split'].eq('retrieval').sum())}")
    print(f"test_images={len(test_df)}")
    print(test_df["disease_category"].value_counts().sort_index().to_string())
    print(f"patient_overlap_between_splits={len(patient_overlap)}")


if __name__ == "__main__":
    main()
