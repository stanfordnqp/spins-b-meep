import logging

import numpy as np
import pytest
import scipy.sparse

from spins import goos
from spins.goos_sim import meep

logging.basicConfig(format=goos.LOG_FORMAT)
logging.getLogger("").setLevel(logging.DEBUG)


def test_simulate_wg_straight_2d():
    with goos.OptimizationPlan() as plan:
        eps = goos.Cuboid(pos=goos.Constant([0, 0, 0]),
                          extents=goos.Constant([6000, 500, 40]),
                          material=goos.material.Material(index=3.45))
        sim = meep.FdtdSimulation(
            eps=eps,
            sources=[
                meep.WaveguideModeSource(
                    center=[-500, 0, 0],
                    extents=[0, 2500, 0],
                    normal=[1, 0, 0],
                    mode_num=0,
                    power=1,
                    wavelength=1550,
                    bandwidth=100,
                ),
            ],
            sim_space=meep.SimulationSpace(
                dx=20,
                sim_region=goos.Box3d(
                    center=[0, 0, 0],
                    extents=[4000, 4000, 0],
                ),
                pml_thickness=[800, 800, 800, 800, 0, 0]),
            sim_timing=meep.SimulationTiming(stopping_conditions=[
                meep.StopWhenFieldsDecayed(
                    time_increment=50,
                    component=2,
                    pos=[0, 0, 0],
                    threshold=1e-6,
                )
            ], ),
            background=goos.material.Material(index=1.0),
            outputs=[
                meep.Epsilon(wavelength=1550),
                meep.ElectricField(wavelength=1550),
                meep.WaveguideModeOverlap(wavelength=1550,
                                          center=[1000, 0, 0],
                                          extents=[0, 2500, 0],
                                          normal=[1, 0, 0],
                                          mode_num=0,
                                          power=1),
                meep.WaveguideModeOverlap(wavelength=1550,
                                          center=[-1000, 0, 0],
                                          extents=[0, 2500, 0],
                                          normal=[-1, 0, 0],
                                          mode_num=0,
                                          power=1),
                meep.WaveguideModeOverlap(wavelength=1550,
                                          center=[0, 0, 0],
                                          extents=[0, 2500, 0],
                                          normal=[1, 0, 0],
                                          mode_num=0,
                                          power=1),
            ])

        out = sim.get()

        # Plot the fields.
        #import matplotlib.pyplot as plt
        #eps = out[0].array
        #plt.imshow(eps)

        #plt.figure()
        #field = out[1].array
        #plt.imshow(np.real(field[2]))

        #plt.show()

        # Power transmitted should be unity but numerical dispersion could
        # affect the actual transmitted power. Last time we checked, the power
        # ~0.96.
        assert np.abs(out[4].array)**2 >= 0.96

        # Check that waveguide power is roughly constant along waveguide.
        np.testing.assert_allclose(np.abs(out[2].array)**2,
                                   np.abs(out[4].array)**2,
                                   rtol=1e-2)

        # Check that we get minimal leakage of power flowing backwards.
        assert np.abs(out[3].array)**2 / np.abs(out[2].array)**2 < 0.01


def test_simulate_wg_straight_2d_multicore():
    with goos.OptimizationPlan() as plan:
        eps = goos.Cuboid(pos=goos.Constant([0, 0, 0]),
                          extents=goos.Constant([6000, 500, 40]),
                          material=goos.material.Material(index=3.45))
        sim = meep.FdtdSimulation(
            eps=eps,
            sources=[
                meep.WaveguideModeSource(
                    center=[-500, 0, 0],
                    extents=[0, 2500, 0],
                    normal=[1, 0, 0],
                    mode_num=0,
                    power=1,
                    wavelength=1550,
                    bandwidth=100,
                ),
            ],
            sim_space=meep.SimulationSpace(
                dx=20,
                sim_region=goos.Box3d(
                    center=[0, 0, 0],
                    extents=[4000, 4000, 0],
                ),
                pml_thickness=[800, 800, 800, 800, 0, 0]),
            sim_timing=meep.SimulationTiming(stopping_conditions=[
                meep.StopWhenFieldsDecayed(
                    time_increment=50,
                    component=2,
                    pos=[0, 0, 0],
                    threshold=1e-6,
                )
            ]),
            background=goos.material.Material(index=1.0),
            outputs=[
                meep.Epsilon(wavelength=1550),
                meep.ElectricField(wavelength=1550),
                meep.WaveguideModeOverlap(wavelength=1550,
                                          center=[1000, 0, 0],
                                          extents=[0, 2500, 0],
                                          normal=[1, 0, 0],
                                          mode_num=0,
                                          power=1),
                meep.WaveguideModeOverlap(wavelength=1550,
                                          center=[-1000, 0, 0],
                                          extents=[0, 2500, 0],
                                          normal=[-1, 0, 0],
                                          mode_num=0,
                                          power=1),
                meep.WaveguideModeOverlap(wavelength=1550,
                                          center=[0, 0, 0],
                                          extents=[0, 2500, 0],
                                          normal=[1, 0, 0],
                                          mode_num=0,
                                          power=1),
                meep.WaveguideModeOverlap(wavelength=1500,
                                          center=[0, 0, 0],
                                          extents=[0, 2500, 0],
                                          normal=[1, 0, 0],
                                          mode_num=0,
                                          power=1,
                                          normalize=True),
                meep.WaveguideModeOverlap(wavelength=1500,
                                          center=[0, 0, 0],
                                          extents=[0, 2500, 0],
                                          normal=[1, 0, 0],
                                          mode_num=0,
                                          power=1,
                                          normalize=False),
            ],
            sim_cores=2,
        )

        out = sim.get()

        # Plot the fields.
        #import matplotlib.pyplot as plt
        #eps = out[0].array
        #plt.imshow(eps)

        #plt.figure()
        #field = out[1].array
        #plt.imshow(np.real(field[2]))

        #plt.show()

        # Power transmitted should be unity but numerical dispersion could
        # affect the actual transmitted power. Last time we checked, the power
        # ~0.96.
        # 1550 nm overlap.
        assert np.abs(out[4].array)**2 >= 0.96
        assert np.abs(out[4].array)**2 <= 1.04
        # 1525 nm overlap (with normalization)
        assert np.abs(out[5].array)**2 >= 0.96
        assert np.abs(out[5].array)**2 <= 1.04

        # The unnormalized power should be roughly ~30%.
        assert np.abs(out[6].array)**2 < 0.40
        assert np.abs(out[6].array)**2 > 0.20

        # Check that waveguide power is roughly constant along waveguide.
        np.testing.assert_allclose(np.abs(out[2].array)**2,
                                   np.abs(out[4].array)**2,
                                   rtol=1e-2)

        # Check that we get minimal leakage of power flowing backwards.
        assert np.abs(out[3].array)**2 / np.abs(out[2].array)**2 < 0.01


def test_simulate_wg_opt():
    with goos.OptimizationPlan() as plan:
        wg_in = goos.Cuboid(pos=goos.Constant([-2000, 0, 0]),
                            extents=goos.Constant([3000, 800, 40]),
                            material=goos.material.Material(index=3.45))
        wg_out = goos.Cuboid(pos=goos.Constant([2000, 0, 0]),
                             extents=goos.Constant([3000, 800, 40]),
                             material=goos.material.Material(index=3.45))

        def initializer(size):
            # Set the seed immediately before calling `random` to ensure
            # reproducibility.
            np.random.seed(247)
            return np.random.random(size) * 0.1 + np.ones(size) * 0.7

        var, design = goos.pixelated_cont_shape(
            initializer=initializer,
            pos=goos.Constant([0, 0, 0]),
            extents=[1000, 800, 40],
            pixel_size=[40, 40, 40],
            material=goos.material.Material(index=1),
            material2=goos.material.Material(index=3.45),
        )
        eps = goos.GroupShape([wg_in, wg_out, design])
        sim = meep.FdtdSimulation(
            eps=eps,
            sources=[
                meep.WaveguideModeSource(
                    center=[-1000, 0, 0],
                    extents=[0, 2500, 0],
                    normal=[1, 0, 0],
                    mode_num=2,
                    power=1,
                    wavelength=1550,
                    bandwidth=100,
                )
            ],
            sim_space=meep.SimulationSpace(
                dx=40,
                sim_region=goos.Box3d(
                    center=[0, 0, 0],
                    extents=[4000, 4000, 0],
                ),
                pml_thickness=[400, 400, 400, 400, 0, 0]),
            sim_timing=meep.SimulationTiming(stopping_conditions=[
                meep.StopWhenFieldsDecayed(
                    time_increment=50,
                    component=2,
                    pos=[0, 0, 0],
                    threshold=1e-6,
                )
            ], ),
            background=goos.material.Material(index=1.0),
            outputs=[
                meep.Epsilon(wavelength=1550),
                meep.ElectricField(wavelength=1550),
                meep.WaveguideModeOverlap(wavelength=1550,
                                          center=[1000, 0, 0],
                                          extents=[0, 2500, 0],
                                          normal=[1, 0, 0],
                                          mode_num=0,
                                          power=1),
            ])

        obj = -goos.abs(sim[2])

        import meep as mp
        mp.quiet()
        goos.opt.scipy_minimize(obj,
                                "L-BFGS-B",
                                monitor_list=[obj],
                                max_iters=5)
        plan.run()

        # Check that we can optimize. We choose something over 60% as
        # incorrect gradients will typically not reach this point.
        assert obj.get().array < -0.85

        # As a final check, compare simulation results against Maxwell.
        # Note that dx = 40 for Meep is actually too innaccurate. We therefore
        # resimulate the final structure for both Meep and Maxwell.
        from spins.goos_sim import maxwell
        sim_fdfd = maxwell.fdfd_simulation(
            wavelength=1550,
            eps=eps,
            sources=[
                maxwell.WaveguideModeSource(
                    center=[-1000, 0, 0],
                    extents=[0, 2500, 0],
                    normal=[1, 0, 0],
                    mode_num=2,
                    power=1,
                )
            ],
            simulation_space=maxwell.SimulationSpace(
                mesh=maxwell.UniformMesh(dx=20),
                sim_region=goos.Box3d(
                    center=[0, 0, 0],
                    extents=[4000, 4000, 0],
                ),
                pml_thickness=[400, 400, 400, 400, 0, 0]),
            background=goos.material.Material(index=1.0),
            outputs=[
                maxwell.Epsilon(),
                maxwell.ElectricField(),
                maxwell.WaveguideModeOverlap(center=[1000, 0, 0],
                                             extents=[0, 2500, 0],
                                             normal=[1, 0, 0],
                                             mode_num=0,
                                             power=1),
            ],
            solver="local_direct")

        sim_fdtd_hi = meep.FdtdSimulation(
            eps=eps,
            sources=[
                meep.WaveguideModeSource(
                    center=[-1000, 0, 0],
                    extents=[0, 2500, 0],
                    normal=[1, 0, 0],
                    mode_num=2,
                    power=1,
                    wavelength=1550,
                    bandwidth=100,
                )
            ],
            sim_space=meep.SimulationSpace(
                dx=20,
                sim_region=goos.Box3d(
                    center=[0, 0, 0],
                    extents=[4000, 4000, 0],
                ),
                pml_thickness=[400, 400, 400, 400, 0, 0]),
            sim_timing=meep.SimulationTiming(stopping_conditions=[
                meep.StopWhenFieldsDecayed(
                    time_increment=50,
                    component=2,
                    pos=[0, 0, 0],
                    threshold=1e-6,
                )
            ], ),
            background=goos.material.Material(index=1.0),
            outputs=[
                meep.Epsilon(wavelength=1550),
                meep.ElectricField(wavelength=1550),
                meep.WaveguideModeOverlap(wavelength=1550,
                                          center=[1000, 0, 0],
                                          extents=[0, 2500, 0],
                                          normal=[1, 0, 0],
                                          mode_num=0,
                                          power=1),
            ])
        fdtd_hi_power = goos.abs(sim_fdtd_hi[2])**2
        fdfd_power = goos.abs(sim_fdfd[2])**2

        # Check that power is correct within 0.5%.
        assert np.abs(fdfd_power.get().array - fdtd_hi_power.get().array) < 0.005


if __name__ == "__main__":
    test_simulate_wg_opt()
