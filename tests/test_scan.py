"""Test for the pathway-scan GMT reader."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scan_pathways import read_gmt


def test_read_gmt(tmp_path):
    p = tmp_path / "sets.gmt"
    p.write_text("PATH_A\tdesc\tTh\tDrd1\tDrd2\n"
                 "PATH_B\t\tMbp,1.0\tPlp1,1.0\tMog,1.0\n"     # Enrichr-style: empty desc + weights
                 "BAD\tonly_two_fields\n")          # <3 fields -> skipped
    sets = read_gmt(str(p))
    assert sets["PATH_A"] == ["Th", "Drd1", "Drd2"]
    assert sets["PATH_B"] == ["Mbp", "Plp1", "Mog"]
    assert "BAD" not in sets
