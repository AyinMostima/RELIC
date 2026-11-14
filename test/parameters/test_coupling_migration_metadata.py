# Unit tests ensuring coupling YAML files do not carry historical migration metadata.

from pathlib import Path

import pytest

from relic.parameters import get_global_manager, reset_global_manager


class TestCouplingMigrationMetadata:
    # Verify historical migration metadata is absent from coupling parameter YAML files.

    def setup_method(self):
        reset_global_manager()

    def test_coupling_yaml_has_no_historical_metadata(self):
        # All coupling YAMLs should omit migration_from and retired_keys metadata.
        manager = get_global_manager()
        manager.database.cache_dirty = True  # pylint: disable=protected-access
        manager.database.load_all()

        coupling_meta = {
            Path(path): meta
            for path, meta in manager.database._file_metadata.items()  # pylint: disable=protected-access
            if "coupling" in Path(path).parts
        }
        assert coupling_meta, "Coupling metadata registry should not be empty"

        historical = []
        for path, meta in coupling_meta.items():
            if meta.get("migration_from") or meta.get("retired_keys") not in (None, [], {}):
                historical.append(str(path))

        if historical:
            pytest.fail(f"historical migration metadata present in coupling YAML files: {historical}")


