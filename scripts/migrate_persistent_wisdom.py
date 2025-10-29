#!/usr/bin/env python3
"""
One-time migration script: merge persistent_wisdom.md into knowledgebase.md

This script safely migrates existing persistent_wisdom.md content into the
CROSS-EPISODE INSIGHTS section of knowledgebase.md.
"""

import os
import sys
from pathlib import Path


def migrate():
    """Perform the migration."""
    old_file = "persistent_wisdom.md"
    kb_file = "knowledgebase.md"

    # Check if old file exists
    if not os.path.exists(old_file):
        print(f"✓ No {old_file} found - nothing to migrate")
        return True

    print(f"Found {old_file} - beginning migration...")

    # Read old content
    with open(old_file, "r") as f:
        old_wisdom = f.read().strip()

    if not old_wisdom:
        print(f"✓ {old_file} is empty - removing")
        os.remove(old_file)
        return True

    # Read knowledge base
    if os.path.exists(kb_file):
        with open(kb_file, "r") as f:
            kb_content = f.read()
    else:
        print(f"⚠ Warning: {kb_file} not found - creating basic structure")
        kb_content = "# Zork Game World Knowledge Base\n\n"

    # Check if already migrated
    if "## CROSS-EPISODE INSIGHTS" in kb_content:
        print(
            f"⚠ Warning: CROSS-EPISODE INSIGHTS section already exists in {kb_file}"
        )
        print("  Migration may have already occurred.")
        response = input("  Continue anyway? (y/n): ")
        if response.lower() != "y":
            print("Migration cancelled")
            return False

    # Insert cross-episode section
    wisdom_section = f"""## CROSS-EPISODE INSIGHTS

### Migrated from persistent_wisdom.md

{old_wisdom}

"""

    # Insert before map section if it exists, otherwise at end
    if "## CURRENT WORLD MAP" in kb_content:
        map_index = kb_content.find("## CURRENT WORLD MAP")
        kb_content = (
            kb_content[:map_index] + wisdom_section + "\n" + kb_content[map_index:]
        )
    else:
        kb_content = kb_content.rstrip() + "\n\n" + wisdom_section

    # Save updated knowledge base
    with open(kb_file, "w") as f:
        f.write(kb_content)

    print(f"✓ Migrated content to {kb_file}")

    # Backup and remove old file
    backup_path = f"{old_file}.migrated"
    os.rename(old_file, backup_path)
    print(f"✓ Backed up old file to {backup_path}")

    print("\n✓ Migration completed successfully!")
    print(f"  - Cross-episode insights now in {kb_file}")
    print(f"  - Old file backed up to {backup_path}")
    print(
        f"\nYou can safely delete {backup_path} once you've verified the migration."
    )

    return True


if __name__ == "__main__":
    try:
        success = migrate()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
