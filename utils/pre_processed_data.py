from collections import defaultdict
import os
import re
import nltk
from tqdm import tqdm

from utils.files import cache_array, read_cached_array
from utils.settings import _MinimizedFiles, _PreprocessedFiles, settings


class RawDataLoader:
    """Parses raw .txt files into dicts and caches them as .pkl files.
    On subsequent calls, loads directly from the .pkl cache."""

    def __init__(self, raw_files, preprocessed_files:_PreprocessedFiles, minimized_files: _MinimizedFiles):
        self.raw = raw_files
        self.pkl :_PreprocessedFiles= preprocessed_files
        self.min: _MinimizedFiles = minimized_files

    # ------------------------------------------------------------------ #
    # Private parsers (raw .txt → dict)
    # ------------------------------------------------------------------ #

    def _parse_triples(self, raw_fp) -> dict:
        result = defaultdict(list)
        with open(raw_fp, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in tqdm(lines, desc="parsing triples"):
            head, relation, tail = line.strip().split("\t")
            result[head].append((head, relation, tail))
        return result

    def _parse_descriptions(self, raw_fp) -> dict:
        result = {}
        with open(raw_fp, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in tqdm(lines, desc="parsing descriptions"):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                result[parts[0]] = parts[1]
        return result

    def _parse_aliases(self, raw_fp) -> dict:
        result = defaultdict(list)
        with open(raw_fp, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in tqdm(lines, desc="parsing aliases"):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                entity_id = parts[0]
                result[entity_id] = parts[1:]   # first entry is canonical name, rest are aliases
        return result

    def _parse_relations(self, raw_fp) -> dict:
        result = defaultdict(list)
        with open(raw_fp, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in tqdm(lines, desc="parsing relations"):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                result[parts[0]] = parts[1:]
        return result

    # ------------------------------------------------------------------ #
    # Public getters (load pkl if exists, else parse raw and save pkl)
    # ------------------------------------------------------------------ #

    def _get(self, pkl_fp, raw_fp, parser):
        if os.path.isfile(pkl_fp):
            return read_cached_array(pkl_fp)
        if raw_fp is None or parser is None:
            raise ValueError(f"No raw file or parser provided for {pkl_fp}, and cache not found.")
        print(f"Cache not found, parsing from raw: {raw_fp}")
        data = parser(raw_fp)
        cache_array(data, pkl_fp)
        return data

    def _get_minimized(self, pkl_fp):
        if not os.path.isfile(pkl_fp):
            raise FileNotFoundError(f"Minimized cache not found: {pkl_fp}. Run minimize.py first.")
        return read_cached_array(pkl_fp)

    def get_triples_train(self, minimized=False) -> dict:
        if minimized:
            return self._get_minimized(self.min.TRIPLES_TRAIN)
        return self._get(self.pkl.TRIPLES_TRAIN, self.raw.TRIPLES_TRAIN, self._parse_triples)

    def get_triples_valid(self) -> dict:
        return self._get(self.pkl.TRIPLES_VALID, self.raw.TRIPLES_VALID, self._parse_triples)

    def get_triples_test(self) -> dict:
        return self._get(self.pkl.TRIPLES_TEST, self.raw.TRIPLES_TEST, self._parse_triples)

    def get_descriptions(self, minimized=False) -> dict:
        if minimized:
            return self._get_minimized(self.min.DESCRIPTIONS)
        return self._get(self.pkl.DESCRIPTIONS, self.raw.DESCRIPTIONS, self._parse_descriptions)

    def get_aliases(self, minimized=False) -> dict:
        if minimized:
            return self._get_minimized(self.min.ALIASES)
        return self._get(self.pkl.ALIASES, self.raw.ALIASES, self._parse_aliases)

    def get_relations(self, minimized=False) -> dict:
        if minimized:
            return self._get_minimized(self.min.RELATIONS)
        return self._get(self.pkl.RELATIONS, self.raw.RELATIONS, self._parse_relations)

    def get_silver_spans(self, minimized=False) -> dict:
        if minimized:
            return self._get_minimized(self.min.SILVER_SPANS)
        return self._get(self.pkl.SILVER_SPANS, None, None)


    def cache_all(self):
        """Parse and cache every dataset. Skips files already cached."""
        self.get_triples_train()
        self.get_triples_valid()
        self.get_triples_test()
        self.get_descriptions()
        self.get_aliases()
        self.get_relations()


class HelpersData:
    """Loads/creates helper data (stop words, special-char patterns)."""

    def __init__(self, helpers_files):
        self.fp = helpers_files

    def get_strange_chars(self) -> list:
        if os.path.isfile(self.fp.STRANGE_CHARS):
            return read_cached_array(self.fp.STRANGE_CHARS)
        patterns = [(re.compile(p, re.IGNORECASE), r) for p, r in {
            r'[€"©]': "",
            r"[áăắặằẳẵǎâấậầẩẫäǟȧǡạȁàảȃāąåǻḁãǽǣ]": "a",
            r"[ḃḅḇ]": "b",
            r"[ćčçḉĉċ]": "c",
            r"[ďḑḓḋḍḏ]": "d",
            r"[éĕěȩḝêếệềểễḙëėẹȅèẻȇēḗḕęẽḛé]": "e",
            r"[ḟ]": "f",
            r"[ǵğǧģĝġḡ]": "g",
            r"[ḫȟḩĥḧḣḥẖ]": "h",
            r"[íĭǐîïḯi̇ịȉìỉȋīįĩḭı]": "i",
            r"[ǰĵ]": "j",
            r"[ḱǩķḳḵ]": "k",
            r"[ĺľļḽḷḹḻ]": "l",
            r"[ḿṁṃ]": "m",
            r"[ńňņṋṅṇǹṉñ]": "n",
            r"[óŏǒôốộồổỗöȫȯȱọőȍòỏơớợờởỡȏōṓṑǫǭõṍṏȭǿøɔ]": "o",
            r"[ṕṗ]": "p",
            r"[ŕřŗṙṛṝȑȓṟ]": "r",
            r"[śṥšṧşŝșṡẛṣṩ]": "s",
            r"[ťţṱțẗṫṭṯ]": "t",
            r"[úŭǔûṷüǘǚǜǖṳụűȕùủưứựừửữȗūṻųůũṹṵ]": "u",
            r"[ṿṽ]": "v",
            r"[ẃŵẅẇẉẁẘ]": "w",
            r"[ẍẋ]": "x",
            r"[ýŷÿẏỵỳỷȳẙỹy]": "y",
            r"[źžẑżẓẕʐ]": "z",
            r"[&]": "and",
        }.items()]
        cache_array(patterns, self.fp.STRANGE_CHARS)
        return patterns

    def get_stop_words(self) -> set:
        if os.path.isfile(self.fp.STOP_WORDS):
            return read_cached_array(self.fp.STOP_WORDS)
        nltk.download("stopwords")
        stop_words = set(nltk.corpus.stopwords.words("english"))
        cache_array(stop_words, self.fp.STOP_WORDS)
        return stop_words

    def cache_all(self):
        self.get_strange_chars()
        self.get_stop_words()


# Module-level singletons — safe to import anywhere, no side effects on load.
data_loader = RawDataLoader(settings.RAW_FILES, settings.PREPROCESSED_FILES, settings.MINIMIZED_FILES)
helpers_data = HelpersData(settings.HELPERS_FILES)


def check_minimized_files():
    min_files = settings.MINIMIZED_FILES
    missing = [p for p in [min_files.DESCRIPTIONS, min_files.ALIASES, min_files.TRIPLES_TRAIN, min_files.RELATIONS] if not os.path.isfile(p)]
    if missing:
        print("Minimized files not found. Run minimize.py first.")
        for p in missing:
            print(f"  Missing: {p}")
        return False
    return True

def check_preprocessed_files():
    files = settings.PREPROCESSED_FILES
    missing = [p for p in [files.DESCRIPTIONS, files.ALIASES, files.TRIPLES_TRAIN, files.RELATIONS] if not os.path.isfile(p)]
    if missing:
        print("Preprocessed files not found. Run any script with choosing no for minimization to generate them.")
        for p in missing:
            print(f"  Missing: {p}")
        return False
    return True

def check_files(use_minimized=False):
    if use_minimized:
        return check_minimized_files()

    return check_preprocessed_files()

