import hashlib
import os
import logging


def check_folder_mk(folder_path: str):
    """
    Check filesystem folder path for existence or create folder(s) on the path.

    Args:
        folder_path (str): folder path.

        Returns:
            str: Full path.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        logging.getLogger().info("directory %s created", folder_path)
    else:
        logging.getLogger().info("directory %s already exists", folder_path)

def hash_file_name(ext: str | None, meta: object) -> str:
    """
    Create a file name using extension and md5 hash of metadata object.

    Args:
        ext (str): file extension.
        meta (object): file metatdata.

        Returns:
            str: File name with extension.
    """
    safe_name = f"{hashlib.md5(str(meta).encode()).hexdigest()}.{ext}"
    return safe_name