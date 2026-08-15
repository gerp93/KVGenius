"""User-configurable location for KVGenius's chat history SQLite database.

See gerp93/KVG_Standards' db-location-versioning.md: hardcoding a fixed
path means the user can never relocate the database (e.g. into a
cloud-synced folder for backup). This wraps kvg_dblocation, reusing
CACHE_DIR's parent as the data directory — the same convention every other
per-user directory in this app already uses (see config.py) rather than
inventing a second one.
"""
from kvg_dblocation import DbLocation

from .config import CACHE_DIR

# CACHE_DIR is PROJECT_ROOT/data/model_cache; its parent (PROJECT_ROOT/data)
# is where this app already keeps all its other per-user data.
db_location = DbLocation(data_dir=CACHE_DIR.parent, default_filename="chat_history.db")


def get_effective_db_path():
    """The chat_history.db path the app should actually open: a
    user-chosen location if one was set via Settings > Database Location,
    otherwise the default under CACHE_DIR's parent."""
    return db_location.get_effective_db_path()
