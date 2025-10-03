# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from operator import itemgetter
from typing import TYPE_CHECKING, Any

from airflow.exceptions import AirflowException
from airflow.providers.common.sql.operators.sql import _convert_to_float_if_possible
from airflow.providers.common.sql.hooks.sql import DbApiHook
from airflow.providers.common.sql.version_compat import BaseHook, BaseSensorOperator

if TYPE_CHECKING:
    from airflow.utils.context import Context


class SqlSensor(BaseSensorOperator):
    """
    Run a SQL statement repeatedly until a criteria is met.

    This will keep trying until success or failure criteria are met, or if the
    first cell is not either ``0``, ``'0'``, ``''``, or ``None``. Optional
    success and failure callables are called with the first cell returned as the
    argument.

    If success callable is defined, the sensor will keep retrying until the
    criteria is met. If failure callable is defined, and the criteria is met,
    the sensor will raise AirflowException. Failure criteria is evaluated before
    success criteria. A fail_on_empty boolean can also be passed to the sensor
    in which case it will fail if no rows have been returned.

    :param conn_id: The connection to run the sensor against
    :param sql: The SQL to run. To pass, it needs to return at least one cell
        that contains a non-zero / empty string value.
    :param parameters: The parameters to render the SQL query with (optional).
    :param success: Success criteria for the sensor is a Callable that takes the output
        of selector as the only argument, and returns a boolean (optional).
    :param failure: Failure criteria for the sensor is a Callable that takes the output
        of selector as the only argument and returns a boolean (optional).
    :param selector: Function which takes the resulting row and transforms it before
        it is passed to success or failure (optional). Takes the first cell by default.
    :param fail_on_empty: Explicitly fail on no rows returned.
    :param hook_params: Extra config params to be passed to the underlying hook.
            Should match the desired hook constructor params.
    """

    template_fields: Sequence[str] = ("sql", "hook_params", "parameters")
    template_ext: Sequence[str] = (".hql", ".sql")
    ui_color = "#7c7287"

    def __init__(
        self,
        *,
        conn_id: str,
        sql: str,
        parameters: Mapping[str, Any] | None = None,
        success: Callable[[Any], bool] | None = None,
        failure: Callable[[Any], bool] | None = None,
        selector: Callable[[tuple[Any]], Any] = itemgetter(0),
        fail_on_empty: bool = False,
        hook_params: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> None:
        self.conn_id = conn_id
        self.sql = sql
        self.parameters = parameters
        self.success = success
        self.failure = failure
        self.selector = selector
        self.fail_on_empty = fail_on_empty
        self.hook_params = hook_params
        super().__init__(**kwargs)

    def _get_hook(self) -> DbApiHook:
        conn = BaseHook.get_connection(self.conn_id)
        hook = conn.get_hook(hook_params=self.hook_params)
        if not isinstance(hook, DbApiHook):
            raise AirflowException(
                f"The connection type is not supported by {self.__class__.__name__}. "
                f"The associated hook should be a subclass of `DbApiHook`. Got {hook.__class__.__name__}"
            )
        return hook

    def poke(self, context: Context) -> bool:
        hook = self._get_hook()

        self.log.info("Poking: %s (with parameters %s)", self.sql, self.parameters)
        records = hook.get_records(self.sql, self.parameters)
        if not records:
            if self.fail_on_empty:
                message = "No rows returned, raising as per fail_on_empty flag"
                raise AirflowException(message)
            return False

        condition = self.selector(records[0])
        if self.failure is not None:
            if callable(self.failure):
                if self.failure(condition):
                    message = f"Failure criteria met. self.failure({condition}) returned True"
                    raise AirflowException(message)
            else:
                message = f"self.failure is present, but not callable -> {self.failure}"
                raise AirflowException(message)

        if self.success is not None:
            if callable(self.success):
                return self.success(condition)
            message = f"self.success is present, but not callable -> {self.success}"
            raise AirflowException(message)
        return bool(condition)


class SQLValueCheckSensor(SqlSensor):
    """Get ``True`` when a SQL query returns a value that matches the expected ``pass_value``.

    :param sql: The SQL to run. The first row is used to perform the check.
    :param conn_id: The connection to run the sensor against.
    :param pass_value: The expected value to compare against the query result.
    :param tolerance: Optional tolerance used for numeric comparisons (e.g. ``0.1`` for 10%).
    :param parameters: Parameters to render the SQL query with (optional).
    :param fail_on_empty: Explicitly fail if the query returns no rows.
    :param hook_params: Extra config params to be passed to the underlying hook. Should match the desired hook
        constructor params.
    """

    template_fields: Sequence[str] = (*SqlSensor.template_fields, "pass_value")
    ui_color = "#7c7287"

    def __init__(
        self,
        *,
        conn_id: str,
        sql: str,
        pass_value: Any,
        tolerance: Any | None = None,
        parameters: Mapping[str, Any] | Sequence[Any] | None = None,
        fail_on_empty: bool = False,
        hook_params: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            conn_id=conn_id,
            sql=sql,
            parameters=parameters,
            fail_on_empty=fail_on_empty,
            hook_params=hook_params,
            **kwargs,
        )
        self.pass_value = str(pass_value)
        tol = _convert_to_float_if_possible(tolerance)
        self.tol = tol if isinstance(tol, float) else None
        self.has_tolerance = self.tol is not None

    def poke(self, context: Context) -> bool:
        hook = self._get_hook()
        self.log.info("Poking SQL value check: %s (with parameters %s)", self.sql, self.parameters)
        records = hook.get_first(self.sql, self.parameters)

        if not records:
            if self.fail_on_empty:
                message = "No rows returned, raising as per fail_on_empty flag"
                raise AirflowException(message)
            self.log.info("No rows returned yet for SQL value check.")
            return False

        row = self._normalize_row(records)
        if not row:
            if self.fail_on_empty:
                raise AirflowException("No rows returned, raising as per fail_on_empty flag")
            self.log.info("Row returned no values for SQL value check. Waiting for data.")
            return False

        passed, error_msg = self._evaluate_row(row)
        if passed:
            self.log.info("SQL value check succeeded.")
            return True

        self.log.info("SQL value check not satisfied yet. %s", error_msg)
        return False

    def _normalize_row(self, records: Any) -> list[Any]:
        if isinstance(records, dict):
            return list(records.values())
        if isinstance(records, (list, tuple)):
            return list(records)
        return [records]

    def _evaluate_row(self, row: Sequence[Any]) -> tuple[bool, str]:
        pass_value_conv = _convert_to_float_if_possible(self.pass_value)
        is_numeric_value_check = isinstance(pass_value_conv, float)

        error_msg = (
            "Test failed.\n"
            f"Pass value:{pass_value_conv}\n"
            f"Tolerance:{f'{self.tol:.1%}' if self.tol is not None else None}\n"
            f"Query:\n{self.sql}\n"
            f"Results:\n{row!s}"
        )

        if not row:
            return False, error_msg

        if is_numeric_value_check:
            try:
                numeric_records = self._to_float(row)
            except (ValueError, TypeError) as err:
                raise AirflowException(f"Converting a result to float failed.\n{error_msg}") from err
            tests = self._get_numeric_matches(numeric_records, pass_value_conv)
        else:
            tests = self._get_string_matches(row, pass_value_conv)

        return all(tests), error_msg

    def _to_float(self, records: Sequence[Any]) -> list[float]:
        return [float(record) for record in records]

    def _get_string_matches(self, records: Sequence[Any], pass_value_conv: str) -> list[bool]:
        return [str(record) == pass_value_conv for record in records]

    def _get_numeric_matches(self, numeric_records: Sequence[float], numeric_pass_value_conv: float) -> list[bool]:
        if self.has_tolerance and self.tol is not None:
            return [
                numeric_pass_value_conv * (1 - self.tol)
                <= record
                <= numeric_pass_value_conv * (1 + self.tol)
                for record in numeric_records
            ]

        return [record == numeric_pass_value_conv for record in numeric_records]
