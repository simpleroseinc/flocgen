import os
import argparse
import math
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, List, Tuple
from pyomo_floc import floc_model
from pyomo.environ import SolverFactory, TransformationFactory

# Globals
# List of valid two-letter state codes
VALID_STATE_CODES = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]
# Scale that pushes the limits of IEEE 754 double precision representation
push_ieee_limit = 1.123456789e290


# Functions
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


def haversine_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate the great-circle distance between two points
    on the Earth given their longitudes and latitudes in degrees.
    """

    (lat1, lon1), (lat2, lon2) = p1, p2
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = 3959 * c  # Multiply by radius of Earth in miles

    return distance


def compute_capacities(
    city_population: Dict[str, int],
    facilities: List[str],
    scale_factor: float,
    ieee_limit: bool,
) -> Dict[str, int]:
    """
    Compute the capacity of a facility based on the population.

    Capacity is calculated as 8000 gallons times the ceiling of (population / 10,000),
    and then depending on the scale factor and ieee_limit flag it is scaled or pushed the limits of IEEE 754 double precision representation or both.

    Args:
    city_population (Dict): Contains the population of each city.
    scale_factor (float): How much we should scale the capacity.
    ieee_limit (bool): Whether we should scale the capacity to push the limits of IEEE 754 double precision representation.

    Returns:
    int: The capacity of production.
    """

    # Lambda to calculate capacity in gallons and scale it if needed push the limits of IEEE 754 double precision representation
    compute_capacity = lambda population: (
        push_ieee_limit * (scale_factor * (8000 * math.ceil(population / 10000)))
        if ieee_limit
        else scale_factor * (8000 * math.ceil(population / 10000))
    )
    # Create capacities dictionary
    capacities = {f: compute_capacity(city_population[f]) for f in facilities}

    return capacities


def compute_production_coeff(
    facilities: List[str],
    customers: List[str],
    num_scenarios: int,
    scale_factor: float,
    ieee_limit: bool,
) -> pd.DataFrame:
    """
    Compute the coefficients for the production variables.
    This is only needed due to scaling as by default the coefficients of the production variables are 1.0.

    Args:
    facilities (List): List of facilities.
    customers (List): List of customers.
    num_scenarios (int): The number of scenarios in the problem.
    scale_factor (float): How much we should scale the capacity.
    ieee_limit (bool): Whether we should scale the capacity to push the limits of IEEE 754 double precision representation.

    Returns:
    pd.DataFrame: The coefficient data frame.
    """

    # Lambda to compute coeff values
    compute_coeff = lambda: (
        push_ieee_limit * scale_factor * 1.0 if ieee_limit else scale_factor * 1.0
    )

    # Generate all combinations of FACILITIES, CUSTOMERS, and SCENARIOS using list comprehension
    data = [
        (facility, customer, f"S{scenario+1}", compute_coeff())
        for facility in facilities
        for customer in customers
        for scenario in range(num_scenarios)
    ]

    # Create DataFrame
    coeff_df = pd.DataFrame(
        data, columns=["FACILITIES", "CUSTOMERS", "SCENARIOS", "production_coeff"]
    )

    # Set the index
    coeff_df.set_index(["FACILITIES", "CUSTOMERS", "SCENARIOS"], inplace=True)

    return coeff_df


def generate_demand_scenarios(
    row: pd.Series, num_scenarios: int, scale_factor: float, ieee_limit: bool
) -> pd.DataFrame:
    """
    Given a row of a dataframe with min and max values for customer demand generate scenarios.

    Args:
    row (pd.Series): The row of the dataframe containing min and max values for customer demand.
    num_scenarios (int): The number of scenarios to generate.

    Returns:
    pd.DataFrame: A dataframe with scenarios for the customer demands
    """

    min_val = row["min_demand"]
    max_val = row["max_demand"]

    if num_scenarios > 1:
        interval_size = (max_val - min_val) // (num_scenarios - 1)
    else:  # If only one scenario is requested then use max_val as the demand (iterator i and interval_size are both zero and have no effect)
        interval_size = 0
        min_val = max_val

    compute_scenario_demand = lambda i: (
        push_ieee_limit * (scale_factor * (min_val + i * interval_size))
        if ieee_limit
        else scale_factor * (min_val + i * interval_size)
    )
    scenarios = [
        {
            "customer": row["customer"],
            "scenario": f"S{i+1}",
            "demand": compute_scenario_demand(i),
        }
        for i in range(num_scenarios)
    ]
    return pd.DataFrame(scenarios)


def compute_demands(
    city_population: Dict[str, int],
    customers: List[str],
    num_scenarios: int,
    scale_factor: float,
    ieee_limit: bool,
) -> pd.DataFrame:
    """
    Compute the demand of a location based on the population.

    Demand is calculated as 8000 gallons times the ceiling of (population / 100,000),
    and then depending on the scaling flag converted to push the limits of IEEE 754 double precision representation.

    Args:
    population (int): The population of the city.
    badly_scaled (boolean): Whether we should scale the demand to push the limits of IEEE 754 double precision representation.

    Returns:
    int: The demand for the product.
    """

    # Compute max and min demand for each customer
    min_demand_multiplier = 0.5
    compute_demand = lambda population: 8000 * math.ceil(population / 100000)
    min_demand = {
        c: (compute_demand(city_population[c]) * min_demand_multiplier)
        for c in customers
    }
    max_demand = {c: (compute_demand(city_population[c])) for c in customers}
    demand_df = pd.DataFrame({"min_demand": min_demand, "max_demand": max_demand})
    demand_df.index.name = "customer"

    # Genereate demand scenarios
    scenario_demand_df = pd.concat(
        [
            generate_demand_scenarios(row, num_scenarios, scale_factor, ieee_limit)
            for _, row in demand_df.reset_index().iterrows()
        ],
        axis=0,
    )
    scenario_demand_df = scenario_demand_df.pivot(
        index="customer", columns="scenario", values="demand"
    )

    return scenario_demand_df


def compute_fixed_cost(
    city_population: Dict[str, int], facilities: List[str]
) -> Dict[str, int]:
    """
    Compute the fixed cost for each facility based on the population. The more populated the more expensive.

    Args:
    city_population (Dict): Contains the population of each city.
    facilities (List): List of facilities.

    Returns:
    Dict: The fixed cost for each facility.
    """

    fixed_cost_multiplier = 42
    fixed_cost = {f: (city_population[f] * fixed_cost_multiplier) for f in facilities}

    return fixed_cost


def compute_variable_cost(
    cities_df: pd.DataFrame,
    facilities: List[str],
    customers: List[str],
    cost_per_distance: float,
) -> pd.DataFrame:
    """
    Compute the variable cost for each customer based on haversine distance.

    Args:
    cities_df (pd.DataFrame): DataFrame containing longitude and latitude of cities.
    facilities (List): List of facilities.
    customers (List): List of customers.
    cost_per_distance (float): Cost per distance of transportation.

    Returns:
    pd.DataFrame: The variable cost for each facility/customer combination.
    """

    coords = {row["city"]: (row["lat"], row["lng"]) for _, row in cities_df.iterrows()}
    variable_cost = pd.DataFrame(
        [
            {
                "facility": facility,
                "customer": customer,
                "distance": cost_per_distance
                * haversine_distance(coords[facility], coords[customer]),
            }
            for facility in facilities
            for customer in customers
        ]
    ).set_index(["facility", "customer"])

    return variable_cost


def prep_data(
    state: str,
    num_facilities: int,
    num_customers: int,
    num_scenarios: int,
    cost_per_distance: float,
    scale_factor: float,
    ieee_limit: bool,
) -> Dict:

    # Load the csv of us cities
    # uscities.csv was obtained from https://simplemaps.com/data/us-cities
    all_cities_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/uscities.csv"))

    # Select state, city, latitude, longitude, and population columns
    all_cities_df = all_cities_df[["state_id", "city", "lat", "lng", "population"]]
    # Filter cities by user provided state id
    cities_df = all_cities_df[all_cities_df["state_id"] == state].copy()
    # Drop duplicates (although this is most likely not necessary)
    cities_df.drop_duplicates(subset="city", keep="first", inplace=True)

    # Check if the total number of cities is sufficient
    total_cities = len(cities_df)
    if total_cities < num_facilities + num_customers:
        raise ValueError(
            f"Not enough cities in the state. Required: {num_facilities + num_customers}, Available: {total_cities}. Choose a different state or decrease number of facilities and locations."
        )

    # Sort cities by population in descending order
    cities_df.sort_values(by="population", ascending=False, inplace=True)

    # Obtain indexing sets FACILITIES and CUSTOMERS
    # Select the top num_facilities cities for facilities
    facilities = list(cities_df.head(num_facilities)["city"])
    # Select the next num_customers cities for customers
    customers = list(
        cities_df.iloc[num_facilities : num_facilities + num_customers]["city"]
    )

    # Compute model parameters
    # Create a dictionary to map city names to their populations used for generating demand
    city_population = cities_df.set_index("city")["population"].to_dict()

    # Compute the capacity for each facility
    # Use the compute_capacity function to fill in the capacity dictionary
    facility_capacity = compute_capacities(
        city_population, facilities, scale_factor, ieee_limit
    )

    # Compute the production coefficients for each facility
    production_coeff = compute_production_coeff(
        facilities, customers, num_scenarios, scale_factor, ieee_limit
    )

    # Compute demand for each customer
    customer_demand_df = compute_demands(
        city_population, customers, num_scenarios, scale_factor, ieee_limit
    )

    # Compute fixed cost for each facility (do with population)
    fixed_cost = compute_fixed_cost(city_population, facilities)

    # Compute the variable cost for each customer based on haversine distance
    variable_cost = compute_variable_cost(
        cities_df, facilities, customers, cost_per_distance
    )

    return {
        "FACILITIES": facilities,
        "CUSTOMERS": customers,
        "SCENARIOS": [f"S{i+1}" for i in range(num_scenarios)],
        "prob": {f"S{i+1}": 1 / num_scenarios for i in range(num_scenarios)},
        "fixed_cost": fixed_cost,
        "variable_cost": variable_cost,
        "facility_capacity": facility_capacity,
        "production_coeff": production_coeff,
        "customer_demand": customer_demand_df,
    }


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Provide arguments for generating facility location problems."
    )
    parser.add_argument(
        "--state", type=str, 
        default="TX", 
        help="Two letter U.S. State id (str)"
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
        "--relax",
        action="store_true",
        help="Relax integrality (default: False).",
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

    # Prepare data which can be loaded into AMPL
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
        TransformationFactory('core.relax_integer_vars').apply_to(model)


    # Write it to an .mps file
    model.write(
        f"pyomo_floc_{state}_{num_facilities}_{num_customers}_{num_scenarios}_{cost_per_distance}_{scale_factor}_ieee_{ieee_limit}_relax_{relax}.mps"
    )


if __name__ == "__main__":
    main()
