# Copyright 2025 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import random
import time

import ase
import numpy as np
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize import BFGS

from mlip.simulation.ase.mlip_ase_calculator import MLIPForceFieldASECalculator
from mlip.simulation.configs.ase_config import ASESimulationConfig
from mlip.simulation.enums import SimulationType
from mlip.simulation.simulation_engine import ForceField, SimulationEngine
from mlip.simulation.temperature_scheduling import get_temperature_schedule

SIMULATION_RANDOM_SEED = 42

logger = logging.getLogger("mlip")


class ASESimulationEngine(SimulationEngine):
    """Simulation engine handling simulations with the ASE backend.

    For MD, the NVT-Langevin algorithm is used
    (see `here <https://wiki.fysik.dtu.dk/ase/ase/md.html#module-ase.md.langevin>`_).
    For energy minimization, the BFGS algorithm is used
    (see `here <https://wiki.fysik.dtu.dk/ase/ase/optimize.html#ase.optimize.BFGS>`_).
    """

    Config = ASESimulationConfig

    def __init__(
        self,
        atoms: ase.Atoms,
        force_field: ForceField,
        config: ASESimulationConfig,
    ) -> None:
        super().__init__(atoms, force_field, config)

    def _initialize(
        self,
        atoms: ase.Atoms,
        force_field: ForceField,
        config: ASESimulationConfig,
    ) -> None:
        logger.debug("Initialization of simulation begins...")
        self._config = config
        self.atoms = atoms
        self.atoms.center()
        positions = atoms.get_positions()
        self._num_atoms = positions.shape[0]
        self.state.atomic_numbers = atoms.numbers

        self._init_box()

        self.model_calculator = MLIPForceFieldASECalculator(
            self.atoms,
            self._config.edge_capacity_multiplier,
            force_field,
        )

        self._temperature_schedule = get_temperature_schedule(
            self._config.temperature_schedule_config, self._config.num_steps
        )

        logger.debug("Initialization of simulation completed.")

    def _init_box(self) -> None:
        if isinstance(self._config.box, float):
            self.atoms.cell = np.eye(3) * self._config.box
            self.atoms.pbc = True
        elif isinstance(self._config.box, list):
            self.atoms.cell = np.diag(np.array(self._config.box))
            self.atoms.pbc = True
        else:
            self.atoms.cell = None
            self.atoms.pbc = False

    def run(self) -> None:
        """See documentation of abstract parent class.
        This runs the simulation using the ASE backend.

        Important: The state of the simulation is updated and the loggers are called
        during this function.
        """

        is_md_simulation = self._config.simulation_type == SimulationType.MD

        logger.info("Starting simulation...")
        self.atoms.calc = self.model_calculator
        random.seed(SIMULATION_RANDOM_SEED)

        if is_md_simulation:
            if self.atoms.get_velocities() is None or np.all(
                self.atoms.get_velocities() == 0.0
            ):
                # Set random velocities according to Maxwell-Boltzmann distribution
                MaxwellBoltzmannDistribution(
                    self.atoms, temperature_K=self._config.temperature_kelvin
                )
                Stationary(self.atoms)
                ZeroRotation(self.atoms)
            dyn = Langevin(
                self.atoms,
                timestep=self._config.timestep_fs * units.fs,
                temperature_K=self._config.temperature_kelvin,
                friction=self._config.friction,
            )
        elif self._config.simulation_type == SimulationType.MINIMIZATION:
            dyn = BFGS(self.atoms, logfile=None)
        else:
            raise NotImplementedError(
                f"{self._config.simulation_type=} not implemented for ASE backend"
            )

        def log_to_console() -> None:
            """Logs info to console."""
            step = dyn.get_number_of_steps()
            compute_time = time.perf_counter() - self.self_start_interval_time
            self._log_to_console(step, compute_time)

        def update_state() -> None:
            """Update the internal SimulationState object"""
            step = dyn.get_number_of_steps()
            compute_time = time.perf_counter() - self.self_start_interval_time
            self._update_state(step, compute_time, is_md_simulation)

        def set_beginning_interval_time() -> None:
            self.self_start_interval_time = time.perf_counter()

        def update_temperature() -> None:
            """Update the temperature if a temperature schedule is given."""
            cur_step = dyn.get_number_of_steps()
            temperature_kelvin = self._temperature_schedule(cur_step)
            dyn.set_temperature(temperature_K=temperature_kelvin)

        dyn.attach(log_to_console, interval=self._config.log_interval)
        dyn.attach(update_state, interval=self._config.snapshot_interval)
        dyn.attach(self._call_loggers, interval=self._config.log_interval)
        # Every self._config.log_interval steps, we log. At the end of this logging, we
        # set the beginning of this new interval in order to calculate total compute
        # time

        if is_md_simulation:
            dyn.attach(update_temperature)

        dyn.attach(set_beginning_interval_time, interval=self._config.log_interval)
        self.self_start_interval_time = time.perf_counter()

        if is_md_simulation:
            dyn.run(self._config.num_steps)
        elif self._config.simulation_type == SimulationType.MINIMIZATION:
            dyn.run(
                steps=self._config.num_steps,
                fmax=self._config.max_force_convergence_threshold,
            )
        logger.info("Simulation completed.")

    def _call_loggers(self) -> None:
        for _logger in self.loggers:
            _logger(self.state)

    def _log_to_console(self, step: int, compute_time: float) -> None:
        """Logs timing information to console via our logger."""
        if step == 0:
            logger.debug(
                "Initialization took %.2f seconds.",
                compute_time,
            )
        else:
            logger.info(
                "Steps %s to %s completed in %.2f seconds.",
                self.state.step,
                step,
                compute_time,
            )

    def _update_state(
        self, step: int, compute_time: float, is_md_simulation: bool
    ) -> None:
        """Update the internal state of the simulation.

        Args:
            step: The current step of the simulation
            compute_time: The time spent in the last interval
            is_md_simulation: Whether the simulation is an MD simulation
        """

        def _concat(current: np.ndarray, new: np.ndarray) -> np.ndarray:
            if current is None:
                return np.expand_dims(new, axis=0)
            return np.concatenate([current, np.expand_dims(new, axis=0)], axis=0)

        self.state.positions = _concat(self.state.positions, self.atoms.get_positions())
        self.state.forces = _concat(
            self.state.forces, self.model_calculator.results["forces"]
        )

        self.state.step = step
        self.state.compute_time_seconds += compute_time

        if is_md_simulation:
            kinetic_energy = self.atoms.get_kinetic_energy()
            temperature = self.atoms.get_temperature()
            velocities = self.atoms.get_velocities()
            self.state.temperature = _concat(
                self.state.temperature, np.array(temperature)
            )
            self.state.kinetic_energy = _concat(
                self.state.kinetic_energy, np.array(kinetic_energy)
            )
            self.state.velocities = _concat(self.state.velocities, np.array(velocities))
