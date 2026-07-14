# itzi-core

`itzi-core` is the computational part of the [Itzi flood model](https://www.itzi.org). It provides the numerical simulation engine for distributed, dynamic flood modelling, including surface flow, hydrology and infiltration, and optional coupling to SWMM drainage networks.

## Direction

The project aspires to become a cloud-native flood model. Its architecture separates the simulation engine from data access through provider interfaces, so applications can plug in their own raster input, raster output, and vector output providers. This makes it possible to run the same computational model with local files, in-memory data, or cloud object storage without coupling the solver to a particular storage format or platform.

## Included Providers

The package already includes:

- An xarray raster input provider for static and time-varying gridded data.
- An Icechunk raster output provider for versioned, appendable simulation results backed by Zarr-compatible storage.
- In-memory input and output providers for programmatic use and testing.
- A CSV vector output provider for drainage-network results in object storage.

The xarray and Icechunk providers are available through the optional `cloud` dependency group.

## Installation

Itzi Core requires Python 3.12 or 3.13.

Install the base package with `pip`:

```bash
pip install itzi-core
```

Or add it to a uv-managed project:

```bash
uv add itzi-core
```

Install cloud providers, including xarray and Icechunk support:

```bash
pip install "itzi-core[cloud]"
```

With uv:

```bash
uv add "itzi-core[cloud]"
```

For development from a source checkout, use [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv pip install -e .
```

## Architecture

Simulations are configured with `SimulationBuilder`. A run combines the numerical core with providers that implement the following interfaces:

- `RasterInputProvider` supplies static or time-varying raster inputs and domain metadata.
- `RasterOutputProvider` records gridded results at simulation timesteps and finalizes derived outputs.
- `VectorOutputProvider` records coupled drainage-network results.

Custom providers can implement these interfaces to integrate another data catalog, file format, service, or object store.

## Development

Run an individual test:

```bash
uv run pytest tests/test_xarray_input.py
```

Format the project:

```bash
uvx ruff format .
```

The numerical kernels include Cython extensions. Rebuild the editable installation after changing a `.pyx` file:

```bash
uv pip install -e .
```

## License

Itzi Core is licensed under the [GNU Lesser General Public License v2.1 or later](LICENSE).

## Links

- [Itzi website](https://www.itzi.org)
- [Source repository](https://github.com/ItziModel/itzi-core)
- [Issue tracker](https://github.com/ItziModel/itzi-core/issues/)
