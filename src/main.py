# main.py
from pyomo.util.model_size import build_model_size_report
from utils import *
from ef import *
from benders import benders_solve
from cb_benders import cb_benders_solve


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Realistic stochastic facility location (floc) problem generator and solver (arguments: mode (solver or generator), state, number of facilities, number of customers, number of scenarios, etc.)."
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
        "--capacity_rule",
        type=str,
        choices=["max", "min", "average", "expected"],
        default="max",
        help="Determine how the capacity threshold for the first-stage constraints should be computed (default: max (i.e. robust solution)).",
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
        choices=["mps", "lp", "nl", "ef", "benders", "cb_benders"],
        default="mps",
        help="Mode of operation: 'mps': generate MPS, 'lp': LP, or 'nl': NL file; 'ef': solve the extensive-form model, 'benders': solve the model with Benders decomposition, 'cb_benders': Benders with callbacks on the master problem (default: mps).",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["gurobi", "highs", "rose"],
        default="highs",
        help="Which solver to use (default highs).",
    )
    parser.add_argument(
        "--solver_threads",
        type=non_negative_int,
        default=0,
        help="Number of threads the solver should use (default 0 i.e. automatic).",
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Output solver logs and model statistics (default: False).",
    )
    args = parser.parse_args()

    # Create variables from arguments
    state = args.state
    num_facilities = args.num_facilities
    num_customers = args.num_customers
    num_scenarios = args.num_scenarios
    capacity_rule = CapacityRule[args.capacity_rule.upper()]
    cost_per_distance = args.cost_per_distance
    scale_factor = args.scale_factor
    ieee_limit = args.ieee_limit
    relax = args.relax
    mode = args.mode
    solver = args.solver
    solver_threads = args.solver_threads
    verbose = args.verbose

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

    # Handle different modes
    obj_val = None
    if mode in ["mps", "lp", "nl"]:
        # Create the Pyomo model
        model = floc_model(data, capacity_rule=capacity_rule)
        # Relax integrality
        if relax:
            TransformationFactory("core.relax_integer_vars").apply_to(model)
        # Print out size info of generated model
        print(f"Generated Model Statistics")
        print(build_model_size_report(model))
        # Write MPS or LP file
        file_name = f"floc_{state}_{num_facilities}_{num_customers}_{num_scenarios}_{cost_per_distance}_{scale_factor}_ieee_{ieee_limit}_relax_{relax}.{mode}"
        file_path = os.path.join(args.output_dir, file_name)
        model.write(file_path)
        print(f"Model written to:\n\t{file_path}")
    elif mode == "ef":
        # Create the Pyomo model
        model = floc_model(data, capacity_rule=capacity_rule)
        # Relax integrality
        if relax:
            TransformationFactory("core.relax_integer_vars").apply_to(model)
        s = get_solver(solver)
        solve_model(model, s, solver_threads=solver_threads, verbose=verbose)
        print(
            f"Statistics:\n\tEF Model: {model.nconstraints()} cons, {model.nvariables()} vars"
        )
        obj_val = get_objective_value(model)
        print(
            f"Extensive form solve of floc_{state}_{num_facilities}_{num_customers}_{num_scenarios}_{cost_per_distance}_{scale_factor}_ieee_{ieee_limit}_relax_{relax}"
        )
    elif mode == "benders":
        model = benders_solve(
            data,
            capacity_rule=capacity_rule,
            relax=relax,
            solver=solver,
            threads=solver_threads,
            tol=1e-6,
            verbose=verbose,
        )
        obj_val = get_objective_value(model)
        print(
            f"Benders solve of floc_{state}_{num_facilities}_{num_customers}_{num_scenarios}_{cost_per_distance}_{scale_factor}_ieee_{ieee_limit}_relax_{relax}"
        )
    else:
        model = cb_benders_solve(
            data,
            capacity_rule=capacity_rule,
            relax=relax,
            solver=solver,
            tol=1e-6,
            verbose=verbose,
        )
        obj_val = get_objective_value(model)
        print(
            f"Callback Benders solve of floc_{state}_{num_facilities}_{num_customers}_{num_scenarios}_{cost_per_distance}_{scale_factor}_ieee_{ieee_limit}_relax_{relax}"
        )

    if obj_val is not None:
        open_facilities = [
            f"{i}, {state}"
            for i in model.FACILITIES
            if value(model.facility_open[i]) > 0.5
        ]
        print(f"Open facilities: {open_facilities}")
        print(
            f"Objective: {get_objective_value(model)} (mode: {mode}, solver: {solver})"
        )


if __name__ == "__main__":
    main()
