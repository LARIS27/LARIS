from importlib.metadata import version


def test_package_version_is_available() -> None:
    assert version("laris-onpremise") == "0.1.0"


def test_package_import_is_available() -> None:
    import laris_onpremise

    assert laris_onpremise.__version__ == "0.1.0"


def test_log_linked_command_module_is_packaged() -> None:
    import laris_onpremise.log_linked.command as command
    import laris_onpremise.log_linked.workflow as workflow

    assert callable(command.add_parser)
    assert callable(command.run)
    assert callable(workflow.run_preflight)


def test_evaluation1_reproduction_command_is_packaged() -> None:
    import laris_onpremise.reproduction.evaluation1.command as command
    import laris_onpremise.reproduction.evaluation1.resolver as resolver

    assert callable(command.add_parser)
    assert callable(command.run)
    assert resolver.Evaluation1Release.__name__ == "Evaluation1Release"
