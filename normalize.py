import os
import re
import unicodedata
from tqdm import tqdm

from utils.files import cache_array
from utils.pre_processed_data import data_loader, helpers_data
from utils.settings import settings


VALID_WORD = re.compile(r"^[A-Za-z0-9.,!?;:'\"()-]+$")


def _normalize_text(text, strange_chars):
    for pattern, repl in strange_chars:
        text = pattern.sub(repl, text)
    text = ' '.join(w for w in text.split() if VALID_WORD.match(w))
    text = unicodedata.normalize('NFKC', text)
    return re.sub(r'\s+', ' ', text).strip()


def _normalize_descriptions(descs, strange_chars):
    return {
        k: _normalize_text(v, strange_chars)
        for k, v in tqdm(descs.items(), desc="normalizing descriptions")
    }
    


def _normalize_aliases(aliases, strange_chars, stop_words):
    result = {}
    for k, als_list in tqdm(aliases.items(), desc="normalizing aliases"):
        local = set()
        for als in als_list:
            aa = _normalize_text(als, strange_chars).lower()
            if aa and als not in stop_words:
                local.add(aa)
        result[k] = list(local)
    return result


def _check_minimized_files():
    min_files = settings.MINIMIZED_FILES
    missing = [p for p in [min_files.DESCRIPTIONS, min_files.ALIASES, min_files.TRIPLES_TRAIN, min_files.RELATIONS] if not os.path.isfile(p)]
    if missing:
        print("Minimized files not found. Run minimize.py first.")
        for p in missing:
            print(f"  Missing: {p}")
        return False
    return True


def normalize():
    answer = input("Normalize minimized dataset? [Y/n]: ").strip().lower()
    use_minimized = answer != 'n'

    if use_minimized and not _check_minimized_files():
        return

    if use_minimized:
        out_descs = settings.MINIMIZED_FILES.DESCRIPTIONS
        out_aliases = settings.MINIMIZED_FILES.ALIASES
    else:
        out_descs = settings.PREPROCESSED_FILES.DESCRIPTIONS
        out_aliases = settings.PREPROCESSED_FILES.ALIASES

    print(f"\nWarning: the following files will be overwritten:")
    print(f"  {out_descs}")
    print(f"  {out_aliases}")
    confirm = input("Proceed? [y/n]: ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return

    strange_chars = helpers_data.get_strange_chars()
    stop_words = helpers_data.get_stop_words()

    # ── Descriptions ──────────────────────────────────────────────────────
    descs = data_loader.get_descriptions(minimized=use_minimized)
    norm_descs = _normalize_descriptions(descs, strange_chars)
    del descs
    cache_array(norm_descs, out_descs)
    print(f"  Descriptions saved : {len(norm_descs):,}  → {out_descs}")
    del norm_descs

    # ── Aliases ───────────────────────────────────────────────────────────
    aliases = data_loader.get_aliases(minimized=use_minimized)
    norm_aliases = _normalize_aliases(aliases, strange_chars, stop_words)
    del aliases
    cache_array(norm_aliases, out_aliases)
    print(f"  Aliases saved      : {len(norm_aliases):,}  → {out_aliases}")

    dataset_label = "minimized" if use_minimized else "full"
    print(f"\nDone. Normalized {dataset_label} dataset.")


if __name__ == "__main__":
    normalize()
