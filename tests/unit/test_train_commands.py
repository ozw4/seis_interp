from __future__ import annotations

import pytest

from seis_interp.cli import main


@pytest.mark.parametrize("method", ["ccnet5d", "relational-trace-graph"])
def test_partition_training_is_not_a_cli_command(capsys, method):
    with pytest.raises(SystemExit) as error:
        main(["train", method])
    assert error.value.code == 2
    assert f"invalid choice: '{method}'" in capsys.readouterr().err
