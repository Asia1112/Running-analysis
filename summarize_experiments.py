"""Combine one exported frame inventory per recording into descriptive study tables."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.knee_angle_validation.io_utils import write_csv
from src.knee_angle_validation.provenance import new_analysis_id, recording_metadata
from src.knee_angle_validation.statistics import grouped_availability, grouped_reference_statistics, recording_statistics
from src.knee_angle_validation.validation import JOIN_KEYS


def combine_inventories(paths: list[str | Path]) -> pd.DataFrame:
    """Reject repeated recordings and inconsistent exported availability flags."""
    frames, seen_sources, seen_analyses = [], set(), set()
    flags = ("in_extraction", "has_reference_row", "selected_for_reference", "selection_provided",
             "reference_scope", "mediapipe_available", "reference_available", "paired_available")
    for path in paths:
        data = pd.read_csv(path, dtype={key: str for key in ("analysis_id", "source_sha256", "video_id", "person_id", "side")})
        required = {*JOIN_KEYS, *flags, "angle_mediapipe", "angle_reference", "frame_quality", "annotator_id", "annotation_session"}
        if data.empty or not required.issubset(data):
            raise ValueError(f"Niekompletny bilans klatek: {path}.")
        metadata = recording_metadata(data)
        if metadata is None:
            raise ValueError("Zbiorcze porównanie wymaga bilansów nowych analiz z identyfikacją filmu SHA-256.")
        for key in ('annotator_id', 'annotation_session'):
            if data[key].isna().any() or data[key].nunique() != 1:
                raise ValueError("Bilans musi dotyczyć jednego anotatora i jednej sesji referencyjnej.")
        if 'speed_kmh' in data:
            speed = pd.to_numeric(data.speed_kmh, errors='coerce')
            if (data.speed_kmh.notna() & (~np.isfinite(speed) | speed.le(0))).any():
                raise ValueError("Nieprawidłowa prędkość w bilansie.")
            data['speed_kmh'] = speed
        if metadata['source_sha256'] in seen_sources or metadata['analysis_id'] in seen_analyses:
            raise ValueError("Powtórzone nagranie lub analiza. Wybierz jedną analizę i jedną sesję referencyjną na film.")
        seen_sources.add(metadata['source_sha256'])
        seen_analyses.add(metadata['analysis_id'])
        numbers = pd.to_numeric(data.frame_number, errors='coerce')
        if not np.isfinite(numbers).all() or (numbers < 0).any() or (numbers % 1 != 0).any() or data.duplicated(JOIN_KEYS).any():
            raise ValueError("Nieprawidłowe lub powielone numery klatek w bilansie.")
        for key in flags:
            values = data[key].astype(str).str.lower()
            if not values.isin(['true', 'false']).all():
                raise ValueError(f"Nieprawidłowa flaga bilansu: {key}.")
            data[key] = values.eq('true')
        if data.selection_provided.nunique() != 1:
            raise ValueError("Niespójna informacja o planie anotacji.")
        for key in ('angle_mediapipe', 'angle_reference'):
            data[key] = pd.to_numeric(data[key], errors='coerce')
        expected = {
            'mediapipe_available': data.in_extraction & np.isfinite(data.angle_mediapipe),
            'reference_available': data.has_reference_row & np.isfinite(data.angle_reference),
            'reference_scope': data.selected_for_reference if data.selection_provided.all() else data.has_reference_row,
        }
        expected['paired_available'] = expected['mediapipe_available'] & expected['reference_available']
        if any(not data[key].equals(values) for key, values in expected.items()):
            raise ValueError("Flagi dostępności nie zgadzają się z pomiarami w bilansie.")
        if not data.frame_quality.isin(['CLEAR', 'UNCERTAIN', 'UNMARKABLE', 'UNASSESSED', 'NOT_ANNOTATED']).all():
            raise ValueError("Nieprawidłowa ocena jakości w bilansie.")
        data['error_signed'] = (data.angle_mediapipe - data.angle_reference).where(data.paired_available)
        frames.append(data)
    if not frames:
        raise ValueError("Wskaż przynajmniej jeden bilans klatek.")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventories', nargs='+', required=True)
    parser.add_argument('--output-dir', help='New output directory; existing directories are not overwritten')
    args = parser.parse_args()
    try:
        data = combine_inventories(args.inventories)
        folder = Path(args.output_dir) if args.output_dir else Path('data/results/study_reports') / new_analysis_id()
        folder.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    write_csv(grouped_availability(data, ['video_id', 'person_id', 'speed_kmh']), folder / 'availability_summary.csv')
    write_csv(grouped_reference_statistics(data, ['video_id', 'person_id', 'speed_kmh']), folder / 'summary_statistics.csv')
    write_csv(recording_statistics(data), folder / 'recording_statistics.csv')
    print(f'Reports: {folder}')
    print('Descriptive frame-level errors; frames within recordings/people are dependent. Review per-recording results.')


if __name__ == '__main__':
    main()
