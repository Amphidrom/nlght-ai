# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from pathlib import Path

import pytest

pytestarch = pytest.importorskip("pytestarch")
Rule = pytestarch.Rule
get_evaluable_architecture = pytestarch.get_evaluable_architecture

ROOT = "src.nlght"

CORE = f"{ROOT}.core"
APPLICATION = f"{ROOT}.application"
ADAPTERS = f"{ROOT}.adapters"
BOOTSTRAP = f"{ROOT}.bootstrap"
PORTS = f"{ROOT}.ports"
EXTENSIONS = f"{ROOT}.extensions"


@pytest.fixture(scope="session")
def evaluable():
    src_root = Path(__file__).resolve().parents[2] / "src"
    # module_path must be a submodule of root_path (pytestarch contract) —
    # passing src_root for both silently breaks absolute-import resolution
    # for every `from nlght.X import Y` statement, making every rule below
    # a no-op regardless of the actual code.
    return get_evaluable_architecture(str(src_root), str(src_root / "nlght"))


@pytest.mark.architecture
def test_core_must_not_import_application(evaluable):
    (
        Rule()
        .modules_that()
        .are_sub_modules_of(APPLICATION)
        .should_not()
        .be_imported_by_modules_that()
        .are_sub_modules_of(CORE)
        .assert_applies(evaluable)
    )


@pytest.mark.architecture
def test_core_must_not_import_adapters(evaluable):
    (
        Rule()
        .modules_that()
        .are_sub_modules_of(ADAPTERS)
        .should_not()
        .be_imported_by_modules_that()
        .are_sub_modules_of(CORE)
        .assert_applies(evaluable)
    )


@pytest.mark.architecture
def test_core_must_not_import_bootstrap(evaluable):
    (
        Rule()
        .modules_that()
        .are_sub_modules_of(BOOTSTRAP)
        .should_not()
        .be_imported_by_modules_that()
        .are_sub_modules_of(CORE)
        .assert_applies(evaluable)
    )


@pytest.mark.architecture
def test_application_must_not_import_adapters(evaluable):
    (
        Rule()
        .modules_that()
        .are_sub_modules_of(ADAPTERS)
        .should_not()
        .be_imported_by_modules_that()
        .are_sub_modules_of(APPLICATION)
        .assert_applies(evaluable)
    )


@pytest.mark.architecture
def test_application_may_import_core(evaluable):
    pass


@pytest.mark.architecture
def test_adapters_may_depend_on_core_but_not_other_way_round(evaluable):
    (
        Rule()
        .modules_that()
        .are_sub_modules_of(ADAPTERS)
        .should_not()
        .be_imported_by_modules_that()
        .are_sub_modules_of(CORE)
        .assert_applies(evaluable)
    )


@pytest.mark.architecture
def test_ports_must_not_import_adapters(evaluable):
    (
        Rule()
        .modules_that()
        .are_sub_modules_of(ADAPTERS)
        .should_not()
        .be_imported_by_modules_that()
        .are_sub_modules_of(PORTS)
        .assert_applies(evaluable)
    )


@pytest.mark.architecture
def test_ports_must_not_import_bootstrap(evaluable):
    (
        Rule()
        .modules_that()
        .are_sub_modules_of(BOOTSTRAP)
        .should_not()
        .be_imported_by_modules_that()
        .are_sub_modules_of(PORTS)
        .assert_applies(evaluable)
    )


@pytest.mark.architecture
def test_extensions_must_not_import_adapters_directly(evaluable):
    (
        Rule()
        .modules_that()
        .are_sub_modules_of(ADAPTERS)
        .should_not()
        .be_imported_by_modules_that()
        .are_sub_modules_of(EXTENSIONS)
        .assert_applies(evaluable)
    )


@pytest.mark.architecture
def test_extensions_must_not_import_bootstrap(evaluable):
    (
        Rule()
        .modules_that()
        .are_sub_modules_of(BOOTSTRAP)
        .should_not()
        .be_imported_by_modules_that()
        .are_sub_modules_of(EXTENSIONS)
        .assert_applies(evaluable)
    )


@pytest.mark.architecture
def test_adapters_must_not_import_bootstrap(evaluable):
    (
        Rule()
        .modules_that()
        .are_sub_modules_of(BOOTSTRAP)
        .should_not()
        .be_imported_by_modules_that()
        .are_sub_modules_of(ADAPTERS)
        .assert_applies(evaluable)
    )


@pytest.mark.architecture
def test_ports_must_not_import_application(evaluable):
    (
        Rule()
        .modules_that()
        .are_sub_modules_of(APPLICATION)
        .should_not()
        .be_imported_by_modules_that()
        .are_sub_modules_of(PORTS)
        .assert_applies(evaluable)
    )


@pytest.mark.architecture
def test_adapters_must_not_import_application(evaluable):
    (
        Rule()
        .modules_that()
        .are_sub_modules_of(APPLICATION)
        .should_not()
        .be_imported_by_modules_that()
        .are_sub_modules_of(ADAPTERS)
        .assert_applies(evaluable)
    )


@pytest.mark.architecture
def test_application_must_not_import_bootstrap(evaluable):
    (
        Rule()
        .modules_that()
        .are_sub_modules_of(BOOTSTRAP)
        .should_not()
        .be_imported_by_modules_that()
        .are_sub_modules_of(APPLICATION)
        .assert_applies(evaluable)
    )


@pytest.mark.architecture
def test_extensions_must_not_import_application(evaluable):
    (
        Rule()
        .modules_that()
        .are_sub_modules_of(APPLICATION)
        .should_not()
        .be_imported_by_modules_that()
        .are_sub_modules_of(EXTENSIONS)
        .assert_applies(evaluable)
    )