import random
from collections import defaultdict
from tqdm import tqdm

from utils.settings import settings
from utils.files import cache_array


# ── Streaming helpers (no full dict loaded into RAM) ─────────────────────────

def _scan_triple_heads(triples_fp):
    """Stream triples file → return (set of head IDs, total line count)."""
    head_ids = set()
    total = 0
    with open(triples_fp, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="scanning triple heads"):
            head_ids.add(line.split("\t", 1)[0])
            total += 1
    return head_ids, total


def _scan_candidates(descriptions_fp, head_ids):
    """Stream descriptions file → return (candidate entity_ids list, total line count).
    A candidate is an entity_id that also appears as a triple head."""
    candidates = []
    total = 0
    with open(descriptions_fp, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="scanning descriptions"):
            parts = line.split("\t", 1)
            if len(parts) == 2:
                total += 1
                if parts[0] in head_ids:
                    candidates.append(parts[0])
    return candidates, total


# ── Interactive factor prompt ─────────────────────────────────────────────────

def _ask_factor(prompt="Minimization factor (0 < x ≤ 1): "):
    while True:
        try:
            val = float(input(prompt).strip())
            if 0 < val <= 1:
                return val
            print("  Must be strictly greater than 0 and at most 1.")
        except ValueError:
            print("  Invalid input — enter a decimal number (e.g. 0.01).")


# ── Main ──────────────────────────────────────────────────────────────────────

def minimize():
    raw = settings.RAW_FILES
    raw.validate()

    # ── Phase 1: scan to gather estimates ────────────────────────────────
    print("\nScanning dataset to compute estimates...")
    head_ids, total_triples = _scan_triple_heads(raw.TRIPLES_TRAIN)
    candidates, total_descs = _scan_candidates(raw.DESCRIPTIONS, head_ids)
    del head_ids  # no longer needed, free RAM

    avg_tails_per_head = total_triples / len(candidates) if candidates else 1.0

    print(f"\nDataset overview:")
    print(f"  Total train triples      : {total_triples:,}")
    print(f"  Total descriptions : {total_descs:,}")
    print(f"  Candidate heads    : {len(candidates):,}  (have at least one triple)")
    print(f"  Avg tails per head : {avg_tails_per_head:.1f}")

    # ── Phase 2: interactive factor selection ─────────────────────────────
    while True:
        factor = _ask_factor("\nHow much do you want to keep? (0 < factor ≤ 1): ")
        n_heads = max(1, int(len(candidates) * factor))
        est_tails  = int(n_heads * avg_tails_per_head)
        est_total  = n_heads + est_tails

        print(f"\n  With factor {factor}:")
        print(f"    Head descriptions : ~{n_heads:,}")
        print(f"    Tail descriptions : ~{est_tails:,}")
        print(f"    Total             : ~{est_total:,}  ({est_total / total_descs * 100:.3f}% of original)")

        answer = input("\n  Proceed? [y] / [n] abort / [c] change factor: ").strip().lower()
        if answer == "y":
            break
        if answer == "n":
            print("Aborted.")
            return
        # 'c' or anything else: loop back and ask for a new factor

    # ── Phase 3: sample minimized head entity IDs ─────────────────────────
    minimized_entity_ids = set(random.sample(candidates, n_heads))
    del candidates  # free RAM

    # ── Phase 4: stream triples → keep minimized heads ───────────────────
    print("\nStreaming triples...")
    minimized_triples = defaultdict(list)
    tail_ids = set()
    used_relation_ids = set()
    with open(raw.TRIPLES_TRAIN, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_triples, desc="filtering triples"):
            parts = line.strip().split("\t")
            if len(parts) == 3:
                head, relation, tail = parts
                if head in minimized_entity_ids:
                    minimized_triples[head].append((head, relation, tail))
                    tail_ids.add(tail)
                    used_relation_ids.add(relation)

    # ── Phase 5: stream descriptions → keep heads + tails ────────────────
    print("Streaming descriptions...")
    final_entity_ids = minimized_entity_ids | tail_ids
    minimized_descs = {}
    with open(raw.DESCRIPTIONS, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_descs, desc="filtering descriptions"):
            parts = line.strip().split("\t", 1)
            if len(parts) == 2 and parts[0] in final_entity_ids:
                minimized_descs[parts[0]] = parts[1]

    # ── Phase 6: stream relations → keep used relation IDs ───────────────
    print("Streaming relations...")
    minimized_relations = {}
    with open(raw.RELATIONS, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="filtering relations"):
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[0] in used_relation_ids:
                minimized_relations[parts[0]] = parts[1:]

    # ── Phase 7: stream aliases → keep final entity IDs ──────────────────
    print("Streaming aliases...")
    minimized_aliases = {}
    with open(raw.ALIASES, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="filtering aliases"):
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[0] in final_entity_ids:
                minimized_aliases[parts[0]] = parts[1:]

    # ── Phase 8: save ─────────────────────────────────────────────────────
    min_files = settings.MINIMIZED_FILES
    cache_array(minimized_descs,     min_files.DESCRIPTIONS)
    cache_array(minimized_triples,   min_files.TRIPLES_TRAIN)
    cache_array(minimized_relations, min_files.RELATIONS)
    cache_array(minimized_aliases,   min_files.ALIASES)

    print(f"\nDone. Saved to: {settings.FOLDERS.MINIMIZED_DIR}")
    print(f"  Descriptions  : {len(minimized_descs):,}")
    print(f"  Triple heads  : {len(minimized_triples):,}")
    print(f"  Relations     : {len(minimized_relations):,}")
    print(f"  Aliases       : {len(minimized_aliases):,}")


if __name__ == "__main__":
    minimize()
