# SQL Files

This directory contains the SQL query files, that fetch individual features or feature groups.
They are invoked from the data loader, assembled and validated.

All of them are left-merged on the keys spanned by the index* SQL files which are cross-joined if multiple.
