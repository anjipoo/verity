from pathlib import Path
from urllib.request import urlopen
import hashlib

BASE_URL = "https://raw.githubusercontent.com/jlacko/berka-dataset/master"

FILES = {
    "account.asc": "58d7f50abd72e9b1a5568346f74bb54cd71224ee1db9f09a27d7cac563f38cc6",
    "card.asc": "fc669bde6adf6457d87421c0bfb218e9c384a7032c6accd348d207a405e72109",
    "client.asc": "e435c6b92d246f4f0dfd5e2827469d745c06238714c32b3ffb415eebe794e1a7",
    "disp.asc": "ebd801f77b6d322e8ebc08e52f188e7c8fca539325f85f57f8c73434da9d32d8",
    "district.asc": "7f03cf3b9b82f0fdcc3abdf6cc716f145db8e9875c68e2d2e2f7151e9ecf4df3",
    "loan.asc": "68535f609a254aa7a3f03dd8e27dcb822b532df12a0d6046f0666b8dc0b8ae8e",
    "order.asc": "035930fa6acd2ca42a935e654b21e1bb260248f49b6dc6e7de6351b7c4d56d02",
    "trans.asc": "75ab2f39df9d79d79c5c900de90ddd28248b689f214598ac9fa2ff0f574a70d2",
}

OUTPUT_DIR = Path("data/raw/berka")


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)

    return h.hexdigest()


def download_file(filename, expected_hash):
    url = f"{BASE_URL}/{filename}"
    output_path = OUTPUT_DIR / filename

    print(f"\nDownloading {filename}...")

    try:
        with urlopen(url) as response:
            data = response.read()

        output_path.write_bytes(data)

    except Exception as e:
        print(f"ERROR downloading {filename}: {e}")
        return False

    actual_hash = sha256(output_path)

    if actual_hash != expected_hash:
        print("❌ SHA-256 verification FAILED")
        print(f"Expected: {expected_hash}")
        print(f"Actual:   {actual_hash}")
        output_path.unlink(missing_ok=True)
        return False

    print(f"✓ Downloaded and verified: {filename}")
    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BERKA DATASET DOWNLOADER")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    success = True

    for filename, expected_hash in FILES.items():
        if not download_file(filename, expected_hash):
            success = False

    print("\n" + "=" * 60)

    if success:
        print("✓ ALL BERKA FILES DOWNLOADED AND VERIFIED")
        print("=" * 60)

        print("\nFiles:")
        for filename in FILES:
            path = OUTPUT_DIR / filename
            print(f"  ✓ {path}")

    else:
        print("❌ DOWNLOAD FAILED")
        print("Check your internet connection and try again.")


if __name__ == "__main__":
    main()