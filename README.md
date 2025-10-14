# FeGenetics

A genetic algorithm for optimizing force field parameters using spectroscopic data.

## Overview

FeGenetics uses a genetic algorithm to optimize molecular force field parameters by comparing calculated spectroscopic properties (harmonic frequencies, rotational constants, and fundamental frequencies) against experimental target values.

## Prerequisites

1. **Rust toolchain** (nightly)
   - The project uses Rust nightly features
   - Install from https://rustup.rs/

2. **Spectro executable**
   - Required for calculating spectroscopic properties
   - Must be named `spectro` and placed in the same directory as the compiled binary

## Configuration

The program reads configuration from a `target.toml` file that must be in the same directory as the executable.

### Configuration File Format

```toml
# Target spectroscopic values (frequencies in cm^-1, rotational constants in cm^-1)
harm = [3943.976, 3833.989, 1651.332, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0]  # Harmonic frequencies
rots = [27.655730, 14.5054957, 9.2636424]  # Rotational constants (A, B, C)
fund = [3753.156, 3656.489, 1598.834]  # Fundamental frequencies

# Genetic algorithm parameters
number_atoms = 3
population_size = 2000
tournament_size = 150
mutation_rate = 0.20  # Probability of mutation (0.0 to 1.0)
mutation_strength = 5e-9  # Step size for mutations in force constant units
fitness_threshold = 1.0  # Stop when fitness reaches this value or lower

# Paths (relative to executable directory)
initial_guess = ""  # Leave empty for random initialization, or provide path to initial guess
spectro_path = "./spectro"
spectro_in_path = "./spectro.in"
```

### Configuration Parameters

- **harm**: Harmonic frequencies (target values in cm⁻¹). The array includes the fundamental vibrational modes plus additional harmonic terms. For water (3 atoms), this includes 3 fundamental modes plus 6 additional harmonic terms.
- **rots**: Rotational constants in ABC order (target values in cm⁻¹)
- **fund**: Fundamental frequencies (target values in cm⁻¹). For non-linear molecules: 3N-6 modes; for linear molecules: 3N-5 modes (where N = number of atoms)
- **number_atoms**: Number of atoms in the molecule
- **population_size**: Size of the population for the genetic algorithm (larger = more exploration, slower)
- **tournament_size**: Number of organisms in each tournament selection (affects selection pressure)
- **mutation_rate**: Probability of mutation (0.0 to 1.0). Typical values: 0.1-0.3
- **mutation_strength**: Step size for mutations. This is a dimensionless scaling factor applied to force constants. Start with small values (1e-9 to 1e-8) and adjust based on convergence behavior
- **fitness_threshold**: Fitness value below which the algorithm stops (convergence criterion). Lower = better fit
- **initial_guess**: Path to an initial guess organism (optional, leave empty for random initialization)
- **spectro_path**: Path to the spectro executable
- **spectro_in_path**: Path to the spectro input file (defines molecular geometry and calculation parameters)

## Running the Program

### Water Molecule Example

The repository includes a water molecule example configuration in `src/input/`:

1. **Build the project**:
   ```bash
   cargo build --release
   ```

2. **Set up the runtime directory**:
   ```bash
   cd target/release
   cp ../../src/input/target.toml .
   cp ../../src/input/spectro.in .
   ```

3. **Place the spectro executable**:
   ```bash
   # Copy or link the spectro executable to the current directory
   cp /path/to/spectro .
   # OR
   ln -s /path/to/spectro .
   ```

4. **Run the program**:
   ```bash
   ./fegenetics
   ```

### Using an Initial Guess

To start from a known good organism (like the water test case):

1. Copy the initial guess organism to your runtime directory:
   ```bash
   cp -r ../../tests/water_test ./
   ```

2. Update `target.toml` to point to the initial guess:
   ```toml
   initial_guess = "./water_test"
   ```

3. Run the program as normal

## Output

The program will:
- Print progress every 10 generations
- Show fitness values and number of unfit organisms
- Save the best organism periodically to `best/[generation]/[organism_id]/`
- Stop when fitness reaches the threshold or you interrupt it

Example output:
```
Time taken: 5.2s | Generation 0 | Fitness 123.45 | Best Organism abc-123 | Number Unfit: 50
Time taken: 12.7s | Generation 10 | Fitness 98.23 | Best Organism def-456 | Number Unfit: 35
...
Yes. The superior fighter is clear.
Found solution in generation 47. The organism is xyz-789.
The algorithm took 125 seconds to run.
```

## Project Structure

```
FeGenetics/
├── src/
│   ├── main.rs           # Entry point
│   ├── models/           # Organism and population models
│   ├── traditional/      # Traditional genetic algorithm implementation
│   ├── programs/         # Spectro interface
│   ├── utils/            # Utility functions
│   └── input/            # Example configurations
├── tests/
│   ├── water_test/       # Water molecule test case
│   ├── h2co/             # Formaldehyde test case
│   └── simple/           # Simple test case
└── Cargo.toml
```

## Testing

Run the test suite:
```bash
cargo test
```

**Note**: Some tests require specific file paths and may fail when run from different environments. The tests are designed to validate the algorithm logic and file I/O operations. If you encounter path-related test failures, you can:
- Run individual tests: `cargo test test_name`
- Focus on the core algorithm tests that don't depend on file paths
- Update the hard-coded paths in the test files to match your environment (for development only)

## Troubleshooting

### "No such file or directory" errors

- Ensure `target.toml` is in the same directory as the executable
- Ensure `spectro.in` is in the same directory as the executable
- Ensure the `spectro` executable is present and has execute permissions

### Build errors with proc-macro2

If you encounter build errors related to `proc-macro2`, update it:
```bash
cargo update -p proc-macro2
```

## How It Works

1. **Initialization**: Creates a population of organisms with random force field parameters
2. **Evaluation**: Each organism's parameters are used to calculate spectroscopic properties via the `spectro` program
3. **Fitness**: Fitness is calculated by comparing calculated values to target values
4. **Selection**: Tournament selection chooses parents for mating
5. **Mating**: Parents produce offspring through crossover
6. **Mutation**: Random mutations introduce variation
7. **Iteration**: Process repeats until convergence or interruption

The algorithm optimizes force constants across three derivative levels (second, third, and fourth derivatives) to best match the target spectroscopic data.
