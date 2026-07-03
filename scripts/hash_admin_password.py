"""Generate a bcrypt hash for ADMIN_PASSWORD_HASH.

Usage:
    python -m scripts.hash_admin_password
    # then paste the printed hash into your .env as ADMIN_PASSWORD_HASH=$2b$...

Or set ADMIN_PASSWORD env var to hash non-interactively:
    ADMIN_PASSWORD="my secret" python -m scripts.hash_admin_password
"""
import os
import sys
import getpass

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def main() -> int:
    pw = os.getenv("ADMIN_PASSWORD")
    if pw:
        confirm = pw
    else:
        pw = getpass.getpass("Admin password: ")
        confirm = getpass.getpass("Confirm password: ")
        if pw != confirm:
            print("Passwords do not match.", file=sys.stderr)
            return 1
    if len(pw) < 12:
        print("WARNING: password is shorter than 12 characters.", file=sys.stderr)
    h = pwd_context.hash(pw)
    print()
    print("Add this to your .env:")
    print(f"ADMIN_PASSWORD_HASH={h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
