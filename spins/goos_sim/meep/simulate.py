"""Implements a wrapper for Meep electromagnetic solver.

In order to be more extensible, the code is implemented by separating out
sources, outputs and geometry. For each category, there is a schema and
an implementation class. Specifically,
- for sources: `SimSource` and `SimSourceImpl`
- for outputs: `SimOutput` and `SimOutputImpl`
- for geometries: `goos.ShapeFlow` and `GeometryImpl`
- for stopping conditions: `StopCondition` and `StopConditionImpl`

The mapping from schema to implementation class is handled by the registry
(`SIM_REGISTRY`). To register an implementation, use the `register` function.
For example:

```python
@goos.polymorphic_model()
class MySourceSchema(SimSource):
    ...

@register(MySourceSchema)
class MySourceImpl(SimSourceImpl);
    ...
```
"""
from typing import Dict, List, NamedTuple, Tuple, Union

import itertools
import logging

import meep as mp
import numpy as np
import scipy.sparse

from spins import goos
from spins import gridlock
from spins.goos import schema_registry

logger = logging.getLogger(__name__)

# For sources and outputs.
SIM_REGISTRY = schema_registry.SchemaRegistryStack()
# For geometries.
GEOM_REGISTRY = schema_registry.SchemaRegistryStack()


def register(schema, **kwargs):
    def wrapper(cls):
        if issubclass(schema, goos.Flow):
            name = schema.__name__
            registry = GEOM_REGISTRY
        else:
            name = schema._schema.fields["type"].choices[0]
            registry = SIM_REGISTRY
        registry.register(name, schema, cls, kwargs)
        return cls

    return wrapper


class SimulationError(Exception):
    """This exception is raised whenever an error occurs in the simulation."""


class SimSource(goos.Model):
    pass


class SimSourceImpl:
    """Represents a Meep source."""
    def before_sim(self, sim: mp.Simulation) -> None:
        pass

    def power(self, wlen: float) -> float:
        """Returns the amount of power emitted by the source at a wavelength.

        Args:
            wlen: Wavelength to evaluate source power.

        Returns:
            Nominal power emitted by the source.
        """
        raise NotImplemented()


class SimOutput(goos.Model):
    name = goos.types.StringType(default=None)


class SimOutputImpl:
    """Represents an object that helps produce a simulation output.

    Each `SimOutputImpl` type corresponds to a `SimOutput` schema. Each
    `SimOutputImpl` can have the following hooks:
    - before_sim: Called before the simulation is started.
    - eval: Called to retrieve the output.
    - after_sim: Called after simulation is finished. Object should ensure
        that itself is Picklable after a call to `after_sim`.
    - before_adjoint_sim: Called before the adjoint simulation is started.
    """
    def before_sim(self, sim: mp.Simulation) -> None:
        pass

    def eval(self, sim: mp.Simulation) -> goos.Flow:
        raise NotImplemented("{} cannot be evaluated.".format(type(self)))

    def after_sim(self) -> None:
        pass

    def before_adjoint_sim(self, sim: mp.Simulation,
                           grad_val: goos.Flow.Grad) -> None:
        # If an implementation is not provided, then the gradient must be
        # `None` to indicate that the gradient is not needed.
        if grad_val:
            raise NotImplemented(
                "Cannot differentiate with respect to {}".format(type(self)))


class GeometryImpl:
    def __init__(self, shape: goos.Shape, const_flags: goos.Flow.ConstFlags,
                 wlens: List[float]) -> None:
        self._shape = shape
        self._const_flags = const_flags
        self._wlens = wlens

    def eval(self) -> List[mp.GeometricObject]:
        """Creates the corresponding MEEP geometry objects.

        Returns:
            List of objects to add to MEEP simulation.
        """
        return []

    def grad(self, xywz: Tuple, grad_vals: List[np.ndarray],
             wlens: List[float]) -> goos.Flow.Grad:
        """Computes the gradient of the shape.

        For each wavelength `wlens[i]`, `grad` is given `df/deps_i` where `f`
        is the total objective function and `eps_i` is the permittivity at
        wavelength `wlens[i]`.

        Args:
            xyzw: MEEP xyzw metadata indicating the locations of the gradient
                values.
            grad_vals: List of gradients with respect to permittivity, one for
                each wavelength in `wlens`.
            wlens: Wavelength at which gradient is given.

        Returns:
            Gradient with respect to the shape.
        """
        # If the shape is constant, then we can ignore the gradient calculation.
        # Otherwise, throw an error.
        if not self._const_flags:
            raise NotImplemented()
        return None


class StopCondition(goos.Model):
    pass


class StopConditionImpl:
    """Builds a simulation stopping condition for Meep."""
    def __init__(self, params: StopCondition) -> None:
        self._params = params

    def build(self):
        pass


class SimulationSpace(goos.Model):
    """Defines a simulation space.

    A simulation space contains information regarding the permittivity
    distributions but not the fields, i.e. no information regarding sources
    and wavelengths.

    Attributes:
        name: Name to identify the simulation space. Must be unique.
        dx: Mesh spacing.
        sim_region: Simulation domain. PMLs are added into this region.
        pml_thickness: An array `[x_left, x_right, y_left, y_right, z_left,
            z_right]` where each entry corresponds to the length of the PML
            boundary layer in real units.
    """
    type = goos.ModelNameType("simulation_space")
    dx = goos.types.FloatType()
    sim_region = goos.types.ModelType(goos.Box3d)
    pml_thickness = goos.types.ListType(goos.types.FloatType(),
                                        min_size=6,
                                        max_size=6)


class SimulationTiming(goos.Model):
    """Defines timing information for the simulation.

    Attributes:
        max_timesteps: Maximum number of time steps to run.
        stopping_condition: List of stopping conditions for the simulation.
        until_after_sources: If `True`, stopping conditions are applied after
            sources have ended (see `meep.run` documentation).
    """
    max_timesteps = goos.types.IntType(default=1e8)
    stopping_conditions = goos.types.ListType(
        goos.types.PolyModelType(StopCondition))
    until_after_sources = goos.types.BooleanType(default=True)


def fdtd_simulation(*args, **kwargs):
    return FdtdSimulation(*args, **kwargs)


class FdtdSimulation(goos.ArrayFlowOpMixin, goos.ProblemGraphNode):
    node_type = "sim.meep.fdtd_simulation"

    def __init__(
            self,
            eps: goos.Shape,
            sources: List[SimSource],
            sim_space: SimulationSpace,
            background: goos.material.Material,
            outputs: List[SimOutput],
            sim_timing: SimulationTiming,
            adjoint_sim_timing: SimulationTiming = None,
            sim_cores: int = 1,
    ) -> None:
        # Determine the output flow types.
        output_flow_types = [
            SIM_REGISTRY.get(out.type).meta["output_type"] for out in outputs
        ]
        output_names = [out.name for out in outputs]

        super().__init__([eps],
                         flow_types=output_flow_types,
                         flow_names=output_names)

        # Determine the relevant wavelengths.
        self._wlens = []
        for out in outputs:
            self._wlens.append(out.wavelength)
        self._wlens = list(set(self._wlens))
        self._resolution = 1 / sim_space.dx
        self._sim_region = sim_space.sim_region
        self._pml_thickness = sim_space.pml_thickness

        self._timing = sim_timing
        self._adjoint_timing = adjoint_sim_timing
        if self._adjoint_timing is None:
            self._adjoint_timing = sim_timing

        self._sources = _generate_sources(sources)
        self._outputs = self._generate_outputs(outputs)

        # Maintain list of field monitors as we need them for adjoint.
        # TODO(logansu): We really only need field monitors where the
        # permittivity distribution is non-constant. We can therefore speed
        # up optimization by not capturing the full field.
        self._field_mons = []
        for wlen in self._wlens:
            self._field_mons.append(ElectricField(wavelength=wlen))
        self._field_mons = self._generate_outputs(self._field_mons)
        self._forward_fields = []

        self._last_results = None
        self._last_eps = None

        self._sim_cores = max(sim_cores, 1)

    def eval(self, input_vals: List[goos.Flow],
             context: goos.EvalContext) -> goos.ArrayFlow:
        if not self._last_eps or self._last_eps != input_vals[0]:
            geometry = _generate_geometry(input_vals[0],
                                          context.input_flags[0].const_flags,
                                          self._wlens)

            # Attach field monitors for adjoint.
            outputs = self._field_mons + self._outputs
            sim_args = {
                "wlens": self._wlens,
                "sim_region": self._sim_region,
                "pml_thickness": self._pml_thickness,
                "geometry": geometry.eval(),
                "resolution": self._resolution,
                "sources": self._sources,
                "outputs": outputs,
                "sim_timing": self._timing,
            }
            if self._sim_cores > 1:
                out_vals, outputs = _simulate_parallel(sim_args,
                                                       self._sim_cores)
            else:
                out_vals, outputs = _simulate(**sim_args)

            # Pull out the field monitors from the outputs.
            self._field_mons = outputs[:len(self._field_mons)]
            self._forward_fields = [
                arr.array for arr in out_vals[:len(self._field_mons)]
            ]

            self._outputs = outputs[len(self._field_mons):]
            self._last_results = out_vals[len(self._field_mons):]

            # Cache the simulation.
            self._last_eps = input_vals[0]

        return goos.ArrayFlow(self._last_results)

    def grad(self, input_vals: List[goos.Flow], grad_val: goos.ArrayFlow.Grad,
             context: goos.EvalContext) -> List[goos.Flow.Grad]:
        geometry = _generate_geometry(input_vals[0],
                                      context.input_flags[0].const_flags,
                                      self._wlens)

        sim_args = {
            "wlens": self._wlens,
            "sim_region": self._sim_region,
            "pml_thickness": self._pml_thickness,
            "geometry": geometry.eval(),
            "resolution": self._resolution,
            "sources": [],
            "outputs": self._field_mons,
            "adjoint": True,
            "adjoint_grad_val": grad_val,
            "adjoint_sources": self._outputs,
            "sim_timing": self._adjoint_timing,
        }
        if self._sim_cores > 1:
            adjoint_fields, self._field_mons = _simulate_parallel(
                sim_args, self._sim_cores)
        else:
            adjoint_fields, self._field_mons = _simulate(**sim_args)

        adjoint_fields = [field.array for field in adjoint_fields]

        grad_wrt_eps = []
        for wlen, f1, f2 in zip(self._wlens, self._forward_fields,
                                adjoint_fields):
            grad_wrt_eps.append(2j * np.pi / wlen * f1 * f2)

        # TODO(logansu): Figure out how to handle field coordinates.
        # Get coordinate information.
        #xyzw = adjoint_sim.get_array_metadata(center=self._sim_region.center,
        #                                      size=self._sim_region.extents)
        return [
            geometry.grad(self._field_mons[0]._xyzw + [1 / self._resolution],
                          grad_wrt_eps, self._wlens)
        ]

    def _generate_outputs(self,
                          outputs: List[SimOutput]) -> List[SimOutputImpl]:
        out_impls = []
        for out in outputs:
            cls = SIM_REGISTRY.get(out.type)
            if cls is None:
                raise ValueError("Unsupported output type, got {}".format(
                    out.type))
            out_impls.append(cls.creator(out, self))
        return out_impls

    def get_source_power(self, wlen: float):
        return sum(src.power(wlen) for src in self._sources)


@goos.polymorphic_model()
class WaveguideModeSource(SimSource):
    """Represents a waveguide mode.

    The waveguide is assumed to be axis-aligned. The time-domain profile
    is assumed to be Gaussian centered at wavelength `wavelength` and
    width of `bandwidth`.

    Attributes:
        center: Waveguide center.
        extents: Width and height of waveguide mode region.
        normal: Normal direction of the waveguide. Note that this is also the
            mode propagation direction.
        mode_num: Mode number. The mode with largest propagation constant is
            mode 0, the mode with second largest propagation constant is mode 1,
            etc.
        power: The transmission power of the mode.
        wavelength: Center wavelength of the mode source.
        bandwidth: Bandwidth of the Gaussian pulse.
    """
    type = goos.ModelNameType("source.waveguide_mode")
    center = goos.Vec3d()
    extents = goos.Vec3d()
    normal = goos.Vec3d()
    mode_num = goos.types.IntType()
    power = goos.types.FloatType()

    wavelength = goos.types.FloatType()
    bandwidth = goos.types.FloatType()


@register(WaveguideModeSource)
class WaveguideModeSourceImpl(SimSourceImpl):
    def __init__(self, src: WaveguideModeSource) -> None:
        self._src = src

        # Cache the mode.
        self._wg_mode = None

    def before_sim(self, sim: mp.Simulation) -> None:
        if not self._wg_mode:
            self._wg_mode = _get_waveguide_mode(sim, self._src)

        # Construct source time object on
        for source in _create_waveguide_mode_source(
                src=self._src,
                wg_mode=self._wg_mode,
                src_time=self._get_source_time()):
            sim.add_source(source)

    def power(self, wlen: float) -> float:
        src_time = self._get_source_time()
        # The actual emitted source power is normalized by center frequency.
        amp = src_time.fourier_transform(1 / wlen)
        amp_norm = src_time.fourier_transform(1 / self._src.wavelength)
        return np.abs(amp / amp_norm)**2 * np.sqrt(self._src.power)

    def _get_source_time(self) -> mp.GaussianSource:
        """Constructs the `GaussianSource` object.

        We need to construct this on-the-fly because `GaussianSource` is a SWIG
        object and cannot be Pickled.
        """
        freq = 1 / self._src.wavelength
        freq_hi = 1 / (self._src.wavelength - self._src.bandwidth / 2)
        freq_lo = 1 / (self._src.wavelength + self._src.bandwidth / 2)
        fwidth = (freq_hi - freq_lo) / 2
        # For some reason, `fwidth` is defined in terms of angular frequency
        # rather than frequency...
        fwidth *= 2 * np.pi
        return mp.GaussianSource(frequency=freq, fwidth=fwidth)


@goos.polymorphic_model()
class Epsilon(SimOutput):
    """Displays the permittivity distribution.

    Attributes:
        wavelength: Wavelength to show permittivity.
    """
    type = goos.ModelNameType("output.epsilon")
    wavelength = goos.types.FloatType()


@register(Epsilon, output_type=goos.Function)
class EpsilonImpl(SimOutputImpl):
    def __init__(self, params: Epsilon, sim) -> None:
        self._wlen = params.wavelength

        # Meep will not return data for one of the axis if it is singular,
        # Use this store axis to expand (we ignore 1D sims).
        self._expand_axis = np.nonzero(
            np.array(sim._sim_region.extents) * sim._resolution <= 1)

    def eval(self, sim: mp.Simulation) -> goos.NumericFlow:
        eps = sim.get_epsilon(omega=2 * np.pi / self._wlen)

        if len(eps.shape) < 3:
            eps = np.expand_dims(eps, axis=self._expand_axis)

        return goos.NumericFlow(eps)


@goos.polymorphic_model()
class ElectricField(SimOutput):
    """Retrieves the electric field at a particular wavelength.

    Attributes:
        wavelength: Wavelength to retrieve fields.
    """
    type = goos.ModelNameType("output.electric_field")
    wavelength = goos.types.FloatType()


@register(ElectricField, output_type=goos.Function)
class ElectricFieldImpl(SimOutputImpl):
    def __init__(self, params: ElectricField, sim) -> None:
        self._wlen = params.wavelength
        self._sim_region = sim._sim_region

        # Meep will not return data for one of the axis if it is singular,
        # Use this store axis to expand (we ignore 1D sims).
        self._expand_axis = np.nonzero(
            np.array(sim._sim_region.extents) * sim._resolution <= 1)

    def before_sim(self, sim: mp.Simulation) -> None:
        """Creates the DFT field object used later to get the DFT fields."""
        self._dft_field = sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez],
                                             1 / self._wlen,
                                             1 / self._wlen,
                                             1,
                                             center=self._sim_region.center,
                                             size=self._sim_region.extents)
        # TODO(logansu): Figure out smart way to handle coordinates.
        self._xyzw = sim.get_array_metadata(center=self._sim_region.center,
                                            size=self._sim_region.extents)


    def eval(self, sim: mp.Simulation) -> goos.NumericFlow:
        """Computes frequency-domain fields.

        Frequency domain fields are computed for all three components and
        stacked into one numpy array.

        Returns:
            A `NumericFlow` where the ith entry of the array corresponds to
            electric field component i (e.g. 0th entry is Ex).
        """
        fields = np.stack([
            sim.get_dft_array(self._dft_field, mp.Ex, 0),
            sim.get_dft_array(self._dft_field, mp.Ey, 0),
            sim.get_dft_array(self._dft_field, mp.Ez, 0)
        ],
                          axis=0)

        if len(fields.shape) < 4:
            fields = np.expand_dims(fields, axis=self._expand_axis[0] + 1)
        return goos.NumericFlow(fields)

    def after_sim(self) -> None:
        self._dft_field = None


@goos.polymorphic_model()
class WaveguideModeOverlap(SimOutput):
    """Represents a waveguide mode.

    The waveguide is assumed to be axis-aligned.

    Attributes:
        center: Waveguide center.
        wavelength: Wavelength at which to evaluate overlap.
        extents: Width and height of waveguide mode region.
        normal: Normal direction of the waveguide. Note that this is also the
            mode propagation direction.
        mode_num: Mode number. The mode with largest propagation constant is
            mode 0, the mode with second largest propagation constant is mode 1,
            etc.
        power: The transmission power of the mode.
        normalize: If `True`, normalize the overlap by the square of the total
            power emitted at the `wavelength`.
    """
    type = goos.ModelNameType("overlap.waveguide_mode")
    wavelength = goos.types.FloatType()
    center = goos.Vec3d()
    extents = goos.Vec3d()
    normal = goos.Vec3d()
    mode_num = goos.types.IntType()
    power = goos.types.FloatType()
    normalize = goos.types.BooleanType(default=True)


@register(WaveguideModeOverlap, output_type=goos.Function)
class WaveguideModeOverlapImpl(SimOutputImpl):
    def __init__(self, overlap: WaveguideModeOverlap, sim) -> None:
        if overlap.normalize:
            total_power = sim.get_source_power(overlap.wavelength)
        else:
            total_power = 1

        self._overlap = overlap
        self._amp_factor = 1 / np.sqrt(total_power)
        self._wg_mode = None

    def before_sim(self, sim: mp.Simulation) -> None:
        # Set extents to zero in the direction of the normal as MEEP
        # can automatically deduce the normal this way.
        extents = np.array(self._overlap.extents) * 1.0
        extents[gridlock.axisvec2axis(self._overlap.normal)] = 0

        # Add a flux region to monitor the power.
        self._mon = sim.add_mode_monitor(
            1 / self._overlap.wavelength, 0, 1,
            mp.FluxRegion(center=self._overlap.center, size=extents))

        # Calculate the eigenmode if we have not already.
        if not self._wg_mode:
            self._wg_mode = _get_waveguide_mode(sim, self._overlap)

            norms = _get_overlap_integral(self._wg_mode.mode_fields,
                                          self._wg_mode.mode_fields,
                                          self._wg_mode.xyzw)
            self._mode_norm = np.sqrt(0.5 * np.abs(norms[0] + norms[1]))

    def eval(self, sim: mp.Simulation) -> goos.NumericFlow:

        # Retrieve the relevant tangential fields.
        fields = []
        for comp in self._wg_mode.field_comps:
            fields.append(
                np.reshape(sim.get_dft_array(self._mon, comp, 0),
                           self._wg_mode.xyzw[3].shape))

        norms = _get_overlap_integral(self._wg_mode.mode_fields, fields,
                                      self._wg_mode.xyzw)
        if gridlock.axisvec2polarity(self._overlap.normal) > 0:
            val = 0.5 * (norms[0] + norms[1]) / self._mode_norm
        else:
            val = 0.5 * (norms[0] - norms[1]) / self._mode_norm

        return goos.NumericFlow(val * self._amp_factor)

    def after_sim(self) -> None:
        self._mon = None

    def before_adjoint_sim(self, adjoint_sim: mp.Simulation,
                           grad_val: goos.NumericFlow.Grad) -> None:
        # It turns out that the gradient of the modal overlap turns out to be
        # a TFSF source propagating in the direction opposite of overlap
        # direction (i.e. if the overlap computes power flowing in +x direction
        # then the mode source propagates in -x direction).
        # TODO(logansu): Figure out how to set the appropriate bandwidth
        # for the mode propagating backwards.
        freq = 1 / self._overlap.wavelength
        for src in _create_waveguide_mode_source(
                src=self._overlap,
                wg_mode=self._wg_mode,
                src_time=mp.GaussianSource(frequency=freq, fwidth=0.2 * freq),
                amp_factor=grad_val.array_grad / self._mode_norm * 0.5 *
                self._amp_factor,
                adjoint=True,
        ):
            adjoint_sim.add_source(src)


@goos.polymorphic_model()
class StopWhenFieldsDecayed(StopCondition):
    type = goos.ModelNameType("stop.stop_when_fields_decayed")
    time_increment = goos.types.IntType()
    component = goos.types.IntType()
    pos = goos.Vec3d()
    threshold = goos.types.FloatType()


@register(StopWhenFieldsDecayed)
class StopWhenFieldsDecayedImpl(StopConditionImpl):
    def build(self):
        comps = [mp.Ex, mp.Ey, mp.Ez]
        return mp.stop_when_fields_decayed(self._params.time_increment,
                                           comps[self._params.component],
                                           self._params.pos,
                                           self._params.threshold)


@register(goos.ArrayFlow)
class ArrayImpl(GeometryImpl):
    def __init__(self, shapes: goos.ArrayFlow,
                 const_flags: goos.Flow.ConstFlags,
                 wlens: List[float]) -> None:
        super().__init__(shapes, const_flags, wlens)
        self._shape_impls = []
        for shape, flag in zip(shapes, const_flags.flow_flags):
            self._shape_impls.append(_generate_geometry(shape, flag, wlens))

    def eval(self) -> List[mp.GeometricObject]:
        res = []
        for impl in self._shape_impls:
            res += impl.eval()
        return res

    def grad(self, xyzw: Tuple, grad_vals: List[np.ndarray],
             wlens: List[float]) -> goos.ArrayFlow.Grad:
        return goos.ArrayFlow.Grad(
            [impl.grad(xyzw, grad_vals, wlens) for impl in self._shape_impls])


@register(goos.CuboidFlow)
class CuboidImpl(GeometryImpl):
    def eval(self) -> List[mp.GeometricObject]:
        # TODO(logansu): Do not fix permittivity.
        return [
            mp.Block(
                self._shape.extents,
                center=self._shape.pos,
                material=mp.Medium(
                    epsilon=self._shape.material.permittivity(self._wlens[0])))
        ]


@register(goos.PixelatedContShapeFlow)
class PixelatedContShapeImpl(GeometryImpl):
    """Implements a `PixelatedContShapeFlow`.

    Note that originally we attempted ot use a single `mp.Block` with an
    `epsilon_func`. However, we found that using a single `mp.Block` per pixel
    provided more accurate gradients.
    """
    def eval(self) -> List[mp.GeometricObject]:
        blocks = []
        xcoord, ycoord, zcoord = self._shape.get_cell_coords()
        edge_coords = self._shape.get_edge_coords()
        for i, x in enumerate(xcoord):
            for j, y in enumerate(ycoord):
                for k, z in enumerate(zcoord):
                    val = self._shape.array[i, j, k]
                    eps = (self._shape.material.permittivity(self._wlens[0]) *
                           (1 - val) +
                           self._shape.material2.permittivity(self._wlens[0]) *
                           val)
                    pixel_size = [
                        edge_coords[0][i + 1] - edge_coords[0][i],
                        edge_coords[1][j + 1] - edge_coords[1][j],
                        edge_coords[2][k + 1] - edge_coords[2][k],
                    ]
                    blocks.append(
                        mp.Block(pixel_size,
                                 center=[x, y, z],
                                 material=mp.Medium(epsilon=eps)))
        return blocks

    def grad(self, xyzw: Tuple, grad_vals: List[np.ndarray],
             wlens: List[float]) -> goos.PixelatedContShapeFlow.Grad:
        grad = 0
        for g in grad_vals:
            grad += np.sum(g, axis=0)
        grad = 2 * np.real(grad)

        shape_coords = self._shape.get_edge_coords()
        dx = xyzw[4]
        field_coords = [
            np.r_[np.array(grid) - dx / 2, grid[-1] + dx / 2]
            for grid in xyzw[:3]
        ]
        # TODO(logansu): Correct gradient for dispersive materials.
        contrast = self._shape.material2.permittivity(
            wlens[0]) - self._shape.material.permittivity(wlens[0])
        mat = contrast * get_rendering_matrix(shape_coords, field_coords)
        grad_res = mat.T @ (grad.flatten() * xyzw[3].flatten())
        grad_res = np.reshape(grad_res, self._shape.array.shape)
        return goos.PixelatedContShapeFlow.Grad(array_grad=grad_res)


def get_rendering_matrix(shape_edge_coords, grid_edge_coords):
    mats = [
        get_rendering_matrix_1d(se, re)
        for se, re in zip(shape_edge_coords, grid_edge_coords)
    ]
    return scipy.sparse.kron(scipy.sparse.kron(mats[0], mats[1]), mats[2])


def get_rendering_matrix_1d(shape_coord, grid_coord):
    weights = []
    grid_inds = []
    shape_inds = []

    edge_inds = np.digitize(shape_coord, grid_coord)
    for i, (start_ind, end_ind) in enumerate(zip(edge_inds[:-1],
                                                 edge_inds[1:])):
        # Shape is outside of the first grid cell.
        if end_ind < 1:
            continue
        last_coord = shape_coord[i]
        for j in range(start_ind, end_ind):
            if j >= 1:
                weights.append((grid_coord[j] - last_coord) /
                               (grid_coord[j] - grid_coord[j - 1]))
                grid_inds.append(j - 1)
                shape_inds.append(i)
            last_coord = grid_coord[j]

        if last_coord != shape_coord[i + 1] and end_ind < len(grid_coord):
            weights.append((shape_coord[i + 1] - last_coord) /
                           (grid_coord[end_ind] - grid_coord[end_ind - 1]))
            grid_inds.append(end_ind - 1)
            shape_inds.append(i)
    return scipy.sparse.csr_matrix(
        (weights, (grid_inds, shape_inds)),
        shape=(len(grid_coord) - 1, len(shape_coord) - 1))


def _generate_geometry(shape: Union[goos.ArrayFlow, goos.Shape],
                       const_flags: goos.Flow.ConstFlags,
                       wlens: List[float]) -> GeometryImpl:
    for flow_name, flow_entry in reversed(GEOM_REGISTRY.get_map().items()):
        if isinstance(shape, flow_entry.schema):
            return flow_entry.creator(shape, const_flags, wlens)

    raise ValueError("Encountered unrenderable type, got {}.".format(
        type(shape)))


def _generate_stopping_conds(
        stopping_conds: List[StopCondition]) -> List[StopConditionImpl]:
    impls = []
    if not isinstance(stopping_conds, list):
        stopping_conds = [stopping_conds]

    for cond in stopping_conds:
        cls = SIM_REGISTRY.get(cond.type)
        if cls is None:
            raise ValueError(
                "Encountered unknown stopping condition, got{}".format(
                    cond.type))
        impls.append(cls.creator(cond))
    return impls


class MeepWaveguideMode(NamedTuple):
    """Represents a waveguide mode.

    This is an internal datastructure used to store data for waveguide modes.

    Attributes:
        wlen: Wavelength of the mode.
        field_comps: List of the transverse field components of the waveguide
            mode.
        mode_fields: List of modal fields, one field for each entry in
            `field_comps`.
        xyzw: Mode field metadata. List `(x_coord, y_coord, z_coord, weights)`
            where `x_coord`, `y_coord`, and `z_coord` are the locations at which
            mode fields are sampled and `weights` is the differential weight
            factor used to calculate volume/surface integrals. See Meep
            documentation on `xyzw` for details.
    """
    wlen: float
    field_comps: List
    mode_fields: List[np.ndarray]
    xyzw: List[np.ndarray]


def _get_waveguide_mode(sim: mp.Simulation,
                        src: WaveguideModeSource) -> MeepWaveguideMode:
    """Computes the waveguide mode in Meep (via MPB).

    Args:
        sim: Meep simulation object.
        src: Waveguide mode source properties.

    Returns:
        A tuple `(mode_fields, mode_comps, xyzw)`. `mode_fields`
        is the sampled transverse fields of the mode. `mode_fields` is an array
        with four elements, one for each transverse element as determined by
        `mode_comps`. `xyzw` is the Meep array metadata for the region: It is
        a 4-element tuple `(x, y, z, w)` where the `x`, `y`, and `z` are arrays
        indicating the cell coordinates where the fields are sampled and `w`
        is the volume weight factor used when calculating integrals involving
        the fields (see Meep documentation for details).
    """

    wlen = src.wavelength
    fcen = 1 / wlen
    normal_axis = gridlock.axisvec2axis(src.normal)

    # Extract the eigenmode from Meep.
    # `xyzw` is a tuple `(x_coords, y_coords, z_coords, weights)` where
    # `x_coords`, `y_coords`, and `z_coords` is a list of the coordinates of
    # the Yee cells within the source region.

    # TODO(logansu): Remove this hack when Meep fixes its bugs with
    # `get_array_metadata`. For now, we ensure that the slices are 3D, otherwise
    # the values returned for `w` can be wrong and undeterministic.
    dx = 1 / sim.resolution
    # Count number of dimensions are effectively "zero". This is used to
    # determine the dimensionality of the source (1D or 2D).
    overlap_dims = 3 - sum(val <= dx for val in src.extents)
    extents = [val if val >= dx else dx for val in src.extents]
    xyzw = sim.get_array_metadata(center=src.center, size=extents)

    # TODO(logansu): Understand how to guess a k-vector. Is it necessary?
    k_guess = [0, 0, 0]
    k_guess[normal_axis] = fcen
    k_guess = mp.Vector3(*k_guess)
    mode_dirs = [mp.X, mp.Y, mp.Z]
    mode_data = sim.get_eigenmode(
        fcen, mode_dirs[normal_axis],
        mp.Volume(center=src.center, size=src.extents), src.mode_num + 1,
        k_guess)

    # Determine which field components are relevant for TFSF source (i.e. the
    # field components tangential to the propagation direction.
    # For simplicity, we order them in circular permutation order (x->y->z) with
    # E fields followed by H fields.
    if normal_axis == 0:
        field_comps = [mp.Ey, mp.Ez, mp.Hy, mp.Hz]
    elif normal_axis == 1:
        field_comps = [mp.Ez, mp.Ex, mp.Hz, mp.Hx]
    else:
        field_comps = [mp.Ex, mp.Ey, mp.Hx, mp.Hy]

    # Extract the actual field values for the relevant field components.
    # Note that passing in `mode_data.amplitude` into `amp_func` parameter of
    # `mp.Source` seems to give a bad source.
    mode_fields = []
    for c in field_comps:
        field_slice = []
        for x, y, z in itertools.product(xyzw[0], xyzw[1], xyzw[2]):
            field_slice.append(mode_data.amplitude(mp.Vector3(x, y, z), c))
        field_slice = np.reshape(field_slice,
                                 (len(xyzw[0]), len(xyzw[1]), len(xyzw[2])))
        mode_fields.append(field_slice)

    # Sometimes Meep returns an extra layer of fields when one of the dimensions
    # of the overlap region is zero. In that case, only use the first slice.
    field_slicer = [slice(None), slice(None), slice(None)]
    field_slicer[normal_axis] = slice(0, 1)
    field_slicer = tuple(field_slicer)
    for i in range(4):
        mode_fields[i] = mode_fields[i][field_slicer]

    xyzw[3] = np.reshape(xyzw[3], (len(xyzw[0]), len(xyzw[1]), len(xyzw[2])))
    xyzw[3] = xyzw[3][field_slicer]
    # TODO(logansu): See above TODO about hacking `get_array_metadata`.
    # For now, just guess the correct value. The error introduced by this
    # occurs at the edges of the source/overlap where the fields are small
    # anyway.
    xyzw[3][:] = dx**overlap_dims

    # Fix the phase of the mode by normalizing the phase by the phase where
    # the electric field as largest magnitude.
    arr = np.hstack([mode_fields[0].flatten(), mode_fields[1].flatten()])
    phase_norm = np.angle(arr[np.argmax(np.abs(arr))])
    phase_norm = np.exp(1j * phase_norm)
    mode_fields = [f / phase_norm for f in mode_fields]

    return MeepWaveguideMode(wlen, field_comps, mode_fields, xyzw)


def _get_overlap_integral(field1: List[np.ndarray], field2: List[np.ndarray],
                          xyzw: np.ndarray) -> complex:
    """Computes the overlap integral between two sets of fields.

    Args:
        field1: First field of overlap integral.
        field2: Second field of overlap integral.
        xyzw: Array metadata from Meep.

    Returns:
        The overlap.
    """
    weights = xyzw[3]
    E_overlap = np.sum(
        weights *
        (np.conj(field1[0]) * field2[3] - np.conj(field1[1]) * field2[2]))
    H_overlap = np.sum(
        weights *
        (np.conj(field1[3]) * field2[0] - np.conj(field1[2]) * field2[1]))

    return E_overlap, H_overlap


def _generate_sources(sources: List[SimSource]) -> List[SimSourceImpl]:
    src_list = []
    for src in sources:
        cls = SIM_REGISTRY.get(src.type)
        if cls is None:
            raise ValueError("Unsupported output type, got {}".format(
                out.type))
        src_list.append(cls.creator(src))
    return src_list


def _create_waveguide_mode_source(
        src: Union[WaveguideModeSource, WaveguideModeOverlap],
        wg_mode: MeepWaveguideMode,
        src_time: mp.SourceTime,
        amp_factor: complex = 1,
        adjoint: bool = False,
) -> List[mp.Source]:
    """Creates a TFSF waveguide mode source.

    Roughly speaking, we seek to implement the following
    ```python
    mp.EigenModeSource(
        src=mp.GaussianSource(fcen, fwidth=0.2 * fcen),
        center=src.center,
        size=src.extents,
        eig_match_freq=True,
        eig_band=src.mode_num + 1,
    )
    ```
    The issue is that MEEP seems not to have a way to control the direction
    in which the TFSF source is pointing. To handle this, we instead "manually"
    construct the effective current source ourselves through the following
    steps:
    1. Compute the eigenmode of the waveguide using MEEP.
    2. Extract the field profiles of the mode for the relevant field components
       (i.e. the ones necessary to compute the TFSF source).
    3. Add a `mp.Source` for each component of the TFSF source.

    Args:
        src: Waveguide source parameters.
        wg_mode: Waveguide mode to inject.
        src_time: Specifies the temporal source profile of the source.
        amp_factor: The source amplitude is increased by factor of `amp_factor`.
        adjoint: If `True`, conjugate the modal source.

    Returns:
        A list of MEEP sources corresponding to the TFSF source.
        If `power_wlens` is not `None`, then a list of powers emitted by the
        source is returned as a second return value.
    """
    fcen = 1 / wg_mode.wlen

    # Determine the sign of amplitude of each relevant field component.
    polarity = gridlock.axisvec2polarity(src.normal)
    if adjoint:
        polarity = -polarity
    amp_signs = [1, -1, -polarity, polarity]

    meep_sources = []
    for mode_field, mode_comp, src_comp, amp_sign in zip(
            wg_mode.mode_fields, wg_mode.field_comps,
            reversed(wg_mode.field_comps), amp_signs):
        power_factor = np.sqrt(src.power) / src_time.fourier_transform(fcen)
        if adjoint:
            mode_field = np.conj(mode_field)

        meep_sources.append(
            mp.Source(src=src_time,
                      component=src_comp,
                      center=src.center,
                      size=src.extents,
                      amplitude=amp_sign * amp_factor * power_factor,
                      amp_data=mode_field))
    return meep_sources


def _simulate(
        wlens: List[float],
        sim_region: goos.Box3d,
        pml_thickness: List[float],
        geometry: List[mp.GeometricObject],
        resolution: float,
        sources: List[SimSourceImpl],
        outputs: List[SimOutputImpl],
        sim_timing: SimulationTiming,
        adjoint_grad_val: goos.ArrayFlow.Grad = None,
        adjoint_sources: List[SimOutput] = None,
        adjoint: bool = False,
) -> Tuple[List[goos.Flow], List[SimOutput]]:
    """Runs Meep EM simulation.

    This function factors out the core Meep simulation functionality so that
    we can run the simulation in another process. The idea is to pickle (dill)
    the arguments to this function, which is sent to another process(es) to
    run the simulation.
    """
    # Note that we have to compute PML here because apparently `mp.PML` is
    # a SWIG object, which cannot be pickled.
    pml_layers = []
    for thickness, direction, side in zip(
            pml_thickness, [mp.X, mp.X, mp.Y, mp.Y, mp.Z, mp.Z],
        [mp.Low, mp.High, mp.Low, mp.High, mp.Low, mp.High]):
        pml_layers.append(mp.PML(thickness, direction=direction, side=side))

    sim = mp.Simulation(
        cell_size=sim_region.extents,
        geometry_center=sim_region.center,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=[],
        resolution=resolution,
        force_complex_fields=adjoint,
    )

    # Add the sources. Because the sources may depend on the permittivity
    # distribution, we call `init_sim` to setup permittivity and then call
    # `change_sources` to add the new source list.
    sim.init_sim()

    for src in sources:
        src.before_sim(sim)

    if adjoint:
        for out, g in zip(adjoint_sources, adjoint_grad_val):
            out.before_adjoint_sim(sim, g)

    for out in outputs:
        out.before_sim(sim)

    stop_conds = [sim_timing.max_timesteps] + [
        cond.build()
        for cond in _generate_stopping_conds(sim_timing.stopping_conditions)
    ]
    if sim_timing.until_after_sources:
        sim.run(until_after_sources=stop_conds)
    else:
        sim.run(until=stop_conds)

    results = [out.eval(sim) for out in outputs]

    for out in outputs:
        out.after_sim()

    return results, outputs


def _simulate_parallel(sim_args: Dict, num_cores: int) -> Tuple:
    """Runs a simulation on multiple cores.

    This function launches MPI on `simulate_parallel.py`, which essentially
    just calls `_simulate`. Data is passed by using dill so objects must be
    picklable by dill.

    Args:
        sim_args: List of simulation arguments for `_simulate`. This is directly
            pickled into a temporary file.
        num_cores: Number of cores to run simulation. Sets the number of MPI
            processes to spawn.

    Returns:
        Same as `_simulate`.
    """
    import os
    import pathlib
    import shutil
    import subprocess
    import tempfile

    import dill

    # Create temporary folder for simulation data.
    # We do not use try-finally to delete the temporary directory in case of
    # an error because we want the logs to stick around.
    temp_dir = tempfile.mkdtemp()
    sim_file = os.path.join(temp_dir, "sim.pkl")
    out_file = os.path.join(temp_dir, "sim_out.pkl")

    logger.debug("Dumping simulation to folder %s", temp_dir)

    with open(sim_file, "wb") as fp:
        dill.dump(sim_args, fp)

    # Find the path to the "simulate_parallel.py" file.
    abs_path = pathlib.Path(__file__).parent.absolute()
    script_path = os.path.join(abs_path, "simulate_parallel.py")

    command = [
        "mpirun", "-np",
        str(num_cores), "python", script_path, sim_file, "--outfile", out_file
    ]
    logger.debug("Executing command: %s", " ".join(command))
    with open(os.path.join(temp_dir, "sim.log"), "w") as fp:
        res = subprocess.run(command, stdout=fp, stderr=fp)

    if res.returncode != 0:
        raise SimulationError("MPI returned error, got code {}".format(
            res.returncode))

    with open(out_file, "rb") as fp:
        out_vals, outputs = dill.load(fp)

    # Remove the temporary folder as simulation was successful.
    shutil.rmtree(temp_dir)
    return out_vals, outputs
