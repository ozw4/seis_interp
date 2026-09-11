from __future__ import annotations

import pytest

from seis_interp.cli import main


def test_ccnet5d_partition_training_is_not_a_cli_command(capsys):
    with pytest.raises(SystemExit) as error:
        main(["train", "ccnet5d"])
    assert error.value.code == 2
    assert "invalid choice: 'ccnet5d'" in capsys.readouterr().err
