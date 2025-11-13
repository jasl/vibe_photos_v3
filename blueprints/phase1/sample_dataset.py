#!/usr/bin/env python3
"""Dataset sampling utility for Phase 1 experiments."""

import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class DatasetSampler:
    """Build stratified datasets from a large photo archive."""

    def __init__(self, source_dir: str, target_dir: str) -> None:
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.metadata = {
            "source_path": str(source_dir),
            "target_path": str(target_dir),
            "created_at": datetime.now().isoformat(),
            "samples": [],
        }

    def analyze_dataset(self) -> Dict:
        """Inspect directory structure and aggregate summary statistics."""
        print("📊 Analyzing dataset...")

        stats = {
            "total_dirs": 0,
            "total_files": 0,
            "by_year": defaultdict(int),
            "by_location": defaultdict(int),
            "file_types": defaultdict(int),
            "size_distribution": [],
        }

        for dir_path in self.source_dir.iterdir():
            if not dir_path.is_dir():
                continue

            stats["total_dirs"] += 1

            # Directory names follow the pattern: Location, Month Day, Year
            dir_name = dir_path.name
            year = self._extract_year(dir_name)
            location = self._extract_location(dir_name)

            if year:
                stats["by_year"][year] += 1
            if location:
                stats["by_location"][location] += 1

            # Enumerate files in the folder
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    stats["total_files"] += 1
                    ext = file_path.suffix.lower()
                    stats["file_types"][ext] += 1

                    # Track file sizes for distribution metrics
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    stats["size_distribution"].append(size_mb)

        return stats

    def _extract_year(self, dir_name: str) -> str | None:
        """Parse a four-digit year from a folder name."""
        import re

        match = re.search(r"20\d{2}", dir_name)
        return match.group() if match else None

    def _extract_location(self, dir_name: str) -> str | None:
        """Extract the location token from a directory name."""
        parts = dir_name.split(",")
        if len(parts) > 1:
            return parts[0].strip()
        return None

    def stratified_sample(
        self,
        total_samples: int = 1000,
        strategy: str = "balanced",
    ) -> List[Path]:
        """Perform stratified sampling across the archive.

        Strategies:
        - balanced: evenly sample across available years
        - recent: bias towards the most recent three years
        - random: fully random selection
        """

        print(f"🎲 Running {strategy} sampling with target size {total_samples}")

        all_images: List[Path] = []
        year_groups: Dict[str, List[Path]] = defaultdict(list)

        # Collect all images and group by year
        for dir_path in self.source_dir.iterdir():
            if not dir_path.is_dir():
                continue

            year = self._extract_year(dir_path.name) or "unknown"

            for file_path in dir_path.iterdir():
                if file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".heic"]:
                    all_images.append(file_path)
                    year_groups[year].append(file_path)

        # Sample according to the requested strategy
        if strategy == "balanced":
            samples: List[Path] = []
            years = sorted(year_groups.keys())
            samples_per_year = max(1, total_samples // max(1, len(years)))

            for year in years:
                year_images = year_groups[year]
                n_samples = min(samples_per_year, len(year_images))
                if n_samples:
                    samples.extend(random.sample(year_images, n_samples))

            # Backfill if we undershoot the requested volume
            if len(samples) < total_samples:
                remaining = total_samples - len(samples)
                pool = [img for img in all_images if img not in samples]
                if pool:
                    samples.extend(random.sample(pool, min(remaining, len(pool))))

        elif strategy == "recent":
            recent_years = ["2023", "2024", "2025"]
            recent_images: List[Path] = []
            older_images: List[Path] = []

            for year, images in year_groups.items():
                if year in recent_years:
                    recent_images.extend(images)
                else:
                    older_images.extend(images)

            recent_count = int(total_samples * 0.7)
            older_count = total_samples - recent_count

            samples = []
            if recent_images:
                samples.extend(
                    random.sample(recent_images, min(recent_count, len(recent_images)))
                )
            if older_images:
                samples.extend(
                    random.sample(older_images, min(older_count, len(older_images)))
                )

        else:  # random
            samples = random.sample(all_images, min(total_samples, len(all_images)))

        return samples[:total_samples]

    def create_sample_dataset(
        self, samples: List[Path], preserve_structure: bool = True
    ) -> None:
        """Materialize a curated dataset from sampled paths."""
        print(f"📁 Creating sample dataset with {len(samples)} files")

        self.target_dir.mkdir(parents=True, exist_ok=True)

        for index, source_file in enumerate(samples, 1):
            if preserve_structure:
                # Keep the original directory hierarchy
                rel_dir = source_file.parent.name
                target_subdir = self.target_dir / rel_dir
                target_subdir.mkdir(exist_ok=True)
                target_file = target_subdir / source_file.name
            else:
                # Flatten output into a single directory
                target_file = self.target_dir / f"{index:04d}_{source_file.name}"

            if not target_file.exists():
                # Use symlinks to avoid duplicating storage
                target_file.symlink_to(source_file)
                # Alternatively copy data with: shutil.copy2(source_file, target_file)

            self.metadata["samples"].append(
                {
                    "index": index,
                    "source": str(source_file),
                    "target": str(target_file),
                    "size_mb": source_file.stat().st_size / (1024 * 1024),
                }
            )

            if index % 100 == 0:
                print(f"  Processed {index}/{len(samples)} files...")

        metadata_file = self.target_dir / "dataset_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, indent=2, ensure_ascii=False)

        print(f"✅ Sampling complete. Metadata saved to: {metadata_file}")

    def create_test_sets(self) -> None:
        """Generate multiple datasets for validation and benchmarking."""
        print("🔬 Creating curated test sets...")

        # Phase 1: small validation corpus (1k photos)
        print("\n--- Phase 1: Validation set ---")
        validation_samples = self.stratified_sample(1000, "balanced")
        validation_dir = self.target_dir / "phase1_validation"
        sampler = DatasetSampler(self.source_dir, validation_dir)
        sampler.create_sample_dataset(validation_samples)

        # Phase 2: throughput and performance corpus (5k photos)
        print("\n--- Phase 2: Performance set ---")
        performance_samples = self.stratified_sample(5000, "recent")
        performance_dir = self.target_dir / "phase2_performance"
        sampler = DatasetSampler(self.source_dir, performance_dir)
        sampler.create_sample_dataset(performance_samples)

        # Phase 3: reference to the full archive
        print("\n--- Phase 3: Full archive ---")
        print(f"Source dataset: {self.source_dir}")
        print("30,000+ photos (~400 GB). Use incremental ingestion with resume support.")

    def generate_report(self) -> str:
        """Produce a Markdown summary of the analyzed dataset."""
        stats = self.analyze_dataset()

        report = """# Dataset analysis report

## Overview
"""
        report += f"- Total directories: {stats['total_dirs']:,}\n"
        report += f"- Total files: {stats['total_files']:,}\n"

        report += "\n## Year distribution\n"
        for year in sorted(stats["by_year"].keys()):
            count = stats["by_year"][year]
            report += f"- {year}: {count} directories\n"

        report += "\n## Top 10 locations\n"
        locations = sorted(
            stats["by_location"].items(), key=lambda item: item[1], reverse=True
        )[:10]
        for location, count in locations:
            report += f"- {location}: {count} directories\n"

        report += "\n## File types\n"
        for ext, count in stats["file_types"].items():
            report += f"- {ext}: {count:,} files\n"

        if stats["size_distribution"]:
            avg_size = sum(stats["size_distribution"]) / len(stats["size_distribution"])
            max_size = max(stats["size_distribution"])
            min_size = min(stats["size_distribution"])

            report += "\n## File size distribution\n"
            report += f"- Average: {avg_size:.1f} MB\n"
            report += f"- Largest: {max_size:.1f} MB\n"
            report += f"- Smallest: {min_size:.1f} MB\n"

        return report


def main() -> None:
    """CLI entrypoint for dataset sampling."""
    import argparse

    parser = argparse.ArgumentParser(description="Dataset sampling tool")
    parser.add_argument(
        "--source",
        default="/Users/jasl/Workspaces/exported_photos",
        help="Source dataset directory",
    )
    parser.add_argument(
        "--target",
        default="./test_datasets",
        help="Directory where sampled datasets will be stored",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples to collect",
    )
    parser.add_argument(
        "--strategy",
        choices=["balanced", "recent", "random"],
        default="balanced",
        help="Sampling strategy",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze the dataset without copying files",
    )

    args = parser.parse_args()

    sampler = DatasetSampler(args.source, args.target)

    if args.analyze_only:
        report = sampler.generate_report()
        print(report)

        report_file = Path(args.target) / "dataset_analysis.md"
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as handle:
            handle.write(report)
        print(f"\nReport saved to: {report_file}")
    else:
        samples = sampler.stratified_sample(args.samples, args.strategy)
        sampler.create_sample_dataset(samples)

        report = sampler.generate_report()
        report_file = Path(args.target) / "sampling_report.md"
        with open(report_file, "w", encoding="utf-8") as handle:
            handle.write(report)


if __name__ == "__main__":
    main()
