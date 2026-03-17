

def chunk_dict(d: dict, chunks_n: int):
    """Yield successive n_chunks chunks from a dictionary."""
    chunk_size = (len(d) + chunks_n - 1) // chunks_n 
    items = list(d.items())
    for start in range(0, len(items), chunk_size):
        end = min(start + chunk_size, len(items))
        yield dict(items[start:end])
