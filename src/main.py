# main.py

import argparse
from gen_data import *
from ef import *
from pyomo.environ import *
from pyomo.util.model_size import build_model_size_report
from benders import benders_solve


# Utility functions
def validate_state_code(state_code: str) -> str:
    """
    Validate the state code provided by the user.
    """
    if state_code.upper() not in VALID_STATE_CODES:
        raise argparse.ArgumentTypeError(
            f"Invalid state code: {state_code}. Must be one of {', '.join(VALID_STATE_CODES)}."
        )
    return state_code.upper()


def positive_int(value):
    """
    Validate that the value provided by the user is a positive integer.
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
    return ivalue


def positive_float(value):
    """
    Validate that the value provided by the user is a positive float.
    """
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive float value")
    return fvalue


def get_objective_value(model):
    # Find the active objective(s) in the model
    objs = [obj for obj in model.component_objects(Objective, active=True)]
    if not objs:
        raise RuntimeError("No active objective found in the model.")
    if len(objs) > 1:
        print("Warning: Multiple active objectives found, returning the first one.")
    obj = objs[0]
    return value(obj)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Realistic stochastic facility location (floc) problem generator (arguments: state, number of facilities, number of customers, number of scenarios, etc.)."
    )
    parser.add_argument(
        "--state", type=str, default="TX", help="Two letter U.S. State id (str)"
    )
    parser.add_argument(
        "--num_facilities",
        type=positive_int,
        default=3,
        help="Number of facilities (int)",
    )
    parser.add_argument(
        "--num_customers",
        type=positive_int,
        default=10,
        help="Number of customers (int)",
    )
    parser.add_argument(
        "--num_scenarios",
        type=positive_int,
        default=1,
        help="Number of scenarios (int)",
    )
    parser.add_argument(
        "--cost_per_distance",
        type=positive_float,
        default=1.0,
        help="Cost per distance of transportation (float)",
    )
    parser.add_argument(
        "--scale_factor",
        type=positive_float,
        default=1.0,
        help="Scale factor used to scale capacity and demand parameters (float)",
    )
    parser.add_argument(
        "--ieee_limit",
        action="store_true",
        help="Problem pushed to IEEE 754 representation limits (default: False).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mps", "lp", "ef", "benders"],
        default="mps",
        help="Mode of operation: 'mps': generate MPS, or 'lp': LP file, 'ef': solve the extensive-form model, 'benders': solve the model with Benders decomposition (default: mps).",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["highs", "gurobi"],
        default="highs",
        help="Which solver to use (default highs).",
    )
    parser.add_argument(
        "--relax",
        action="store_true",
        help="Relax integrality (default: False).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory for output files (default: current directory).",
    )
    args = parser.parse_args()

    # Create variables from arguments
    state = args.state
    num_facilities = args.num_facilities
    num_customers = args.num_customers
    num_scenarios = args.num_scenarios
    cost_per_distance = args.cost_per_distance
    scale_factor = args.scale_factor
    ieee_limit = args.ieee_limit
    relax = args.relax
    mode = args.mode
    solver = args.solver

    # Make sure output directory exists (create it if needed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare data which can be loaded into Pyomo
    data = prep_data(
        state,
        num_facilities,
        num_customers,
        num_scenarios,
        cost_per_distance,
        scale_factor,
        ieee_limit,
    )

    # Create the Pyomo model
    model = floc_model(data)

    # Relax integrality
    if relax:
        TransformationFactory("core.relax_integer_vars").apply_to(model)

    # Print out size info of generated model
    print(f"Generated Model Statistics")
    print(build_model_size_report(model))

    # Handle different modes
    result = None
    if mode in ["mps", "lp"]:
        file_name = f"floc_{state}_{num_facilities}_{num_customers}_{num_scenarios}_{cost_per_distance}_{scale_factor}_ieee_{ieee_limit}_relax_{relax}.{mode}"
        file_path = os.path.join(args.output_dir, file_name)
        model.write(file_path)
        # Write file
        print(f"Model written to:\n\t{file_path}")
    elif mode == "ef":
        s = SolverFactory(f"appsi_{solver}")
        s.solve(model)
    else:
        model, history = benders_solve(
            data,
            master_solver=f"{solver}",
            sub_solver=f"{solver}",
            max_iters=50,
            tol=1e-6,
            log=True,
        )

    print(f"Objective: {get_objective_value(model)} (mode: {mode}, solver: {solver})")


if __name__ == "__main__":
    main()
