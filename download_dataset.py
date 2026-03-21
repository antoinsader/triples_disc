import os
import sys
import tarfile
import gzip
import shutil
import urllib.request

from utils.settings import settings

DOWNLOADS = [
    {
        "url": "https://www.dropbox.com/s/7jp4ib8zo3i6m10/wikidata5m_text.txt.gz?dl=1",
        "filename": "wikidata5m_text.txt.gz",
        "type": "gz",
    },
    {
        "url": "https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1",
        "filename": "wikidata5m_alias.tar.gz",
        "type": "tar.gz",
    },
    {
        "url": "https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1",
        "filename": "wikidata5m_transductive.tar.gz",
        "type": "tar.gz",
    },
]


def download_file(url: str, dest_path: str) -> None:
    print(f"Downloading {os.path.basename(dest_path)} ...")

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            pct = min(downloaded / total_size * 100, 100)
            sys.stdout.write(f"\r  {pct:.1f}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
    print()  # newline after progress


def extract_gz(gz_path: str, out_dir: str) -> None:
    out_path = os.path.join(out_dir, os.path.basename(gz_path)[:-3])  # strip .gz
    print(f"Extracting {os.path.basename(gz_path)} -> {os.path.basename(out_path)}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def extract_tar_gz(tar_gz_path: str, out_dir: str) -> None:
    print(f"Extracting {os.path.basename(tar_gz_path)} -> {out_dir}")
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        tar.extractall(path=out_dir)


def main():
    raw_dir = settings.FOLDERS.RAW_DIR
    os.makedirs(raw_dir, exist_ok=True)
    print(f"Target directory: {os.path.abspath(raw_dir)}\n")

    for entry in DOWNLOADS:
        archive_path = os.path.join(raw_dir, entry["filename"])

        download_file(entry["url"], archive_path)

        if entry["type"] == "gz":
            extract_gz(archive_path, raw_dir)
        elif entry["type"] == "tar.gz":
            extract_tar_gz(archive_path, raw_dir)

        os.remove(archive_path)
        print(f"Removed archive {entry['filename']}\n")

    print("All files downloaded and extracted.")


if __name__ == "__main__":
    main()
