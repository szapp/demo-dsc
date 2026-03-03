# Entrypoints

This subpackage holds the interfaces to the outside.

It is the only subpackage that may perform IO operations.

The subpackage is exclusively used externally and must never be imported from other modules.
It is also excluded from tests and coverage.
