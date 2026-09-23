"""Jawne profile eksperymentów publicznego repozytorium."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    label: str
    unet_subject_mode: str
    unet_split_mode: str
    classical_identity_mode: str
    notes: str


PROFILES = {
    "casia": DatasetProfile("casia", "CASIA-IrisV1", "stem_prefix", "per_subject", "auto", "Identyfikator osoby jest początkiem nazwy pliku."),
    "iitd": DatasetProfile("iitd", "IIT Delhi", "parent", "subject_disjoint", "auto", "Identyfikatorem jest domyślnie katalog osoby."),
    "cataract1k": DatasetProfile("cataract1k", "Cataract-1K", "parent", "subject_disjoint", "parent", "Katalog nadrzędny powinien reprezentować nagranie/przypadek."),
}


def profile_for(name: str) -> DatasetProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Nieznana baza: {name}") from exc
