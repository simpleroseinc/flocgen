# Stochastic Facility Location Problem Generator
This repository contains Python scripts that can generate stochastic facility location problems based on U.S. cities.

## TL;DR

- **Install:**
   ```bash
   pip install -r requirements.txt
   ```
- **Usage:** Generate a problem where you try to choose which of 3 facilities to open in the state of Texas given 10 customers, 3 potential levels of future demand, where the transportation cost per distance is 3.14, a scale factor of 1.0 (i.e., **no scaling** is applied to the problem), and we are **not** pushing values to IEEE representation limits:
   ```sh
   cd src/
   python pyomo_gen_floc.py --state TX --num_facilities 3 --num_customers 10 --num_scenarios 3 --cost_per_distance 3.14 --scale_factor 1.0 --ieee_limit False
   ```
- **Output:** A `.mps` file containing the model and data.


## Problem description
Facility location decisions are crucial and often involve significant investment for both public and private sector entities, bearing profound social, economic, and environmental implications. 
The strategic positioning of facilities, such as warehouses, factories, and service centers, can determine an organization's operational efficiency, market reach, and overall sustainability.

Given the high stakes of these decisions, engineers and analysts have developed sophisticated models to aid organizations in identifying optimal locations. 
These models take into account a variety of factors, including but not limited to, transportation costs, proximity to customers and suppliers, labor availability, customer demand, and environmental regulations.

The challenge is compounded when considering the uncertainty inherent in future conditions. 
Factors such as fluctuating market demands, changes in infrastructure, and unpredictable socio-economic developments require a robust approach to facility location. 
Hence, engineers often employ stochastic models and robust optimization techniques that account for such uncertainties, ensuring that the chosen locations remain viable under a range of possible future scenarios.


## Mixed integer program
Below you can find the extensive form of the stochastic facility location problem as an explicit mixed integer program. 

**Given:** 
* A set of facilities: $I$.
* A set of customers: $J$.
* Set of scenarios: $S$ (representing different customer demands).

**Task:** 
* Find the minimum cost facilities to open such that the customer demand can be satisfied in all scenarios.

### Variables
* $x_i \in \{0, 1\} \quad \forall i \in I$
    * $x_i = 1$ if facility $i$ is opened.
* $y_{ij}^s \geq 0 \quad \forall i \in I, \forall j \in J, \forall s \in S$
    * $y_{ij}^s$ is the level of demand for customer $j$ satisfied by facility $i$ in scenario $s$.

### Parameters:
* $\alpha^s$: the probability of scenario $s$.
* $f_i$: the fixed cost for opening facility $i$,
* $q_{ij}$: the cost of servicing customer $j$ from facility $i$,
* $\lambda_j^s$: the demand of customer $j$ in scenario $s$,
* $k_i:$ the capacity of facility $i$.

### The extensive form
The extensive form of our stochastic program can be formulated as follows:

![Equation](https://latex.codecogs.com/svg.image?%5Cbegin%7Bequation%7D%5Cbegin%7Barray%7D%7Brll%7D%5Cmin%5Cquad&%5Csum_%7Bi%5Cin%20I%7Df_i%20x_i&plus;%5Csum_%7Bs%5Cin%20S%7D%5Csum_%7Bi%5Cin%20I%7D%5Csum_%7Bj%5Cin%20J%7D%5Calpha%5Es%20q_%7Bij%7Dy_%7Bij%7D%5Es&%5C%5C&&%5C%5C%5Ctextrm%7Bsubject%20to%7D%5Cquad&%5Csum_%7Bi%5Cin%20I%7Dy_%7Bij%7D%5Es%5Cgeq%5Clambda_j%5Es&%5Cforall%20j%5Cin%20J,%5Cforall%20s%5Cin%20S%5C%5C&%5Csum_%7Bj%5Cin%20J%7Dy_%7Bij%7D%5Es%5Cleq%20k_i%20x_i&%5Cforall%20i%5Cin%20I,%5Cforall%20s%5Cin%20S%5C%5C&%5Csum_%7Bi%5Cin%20I%7Dk_i%20x_i%5Cgeq%5Cmax_%7Bs%5Cin%20S%7D%5Csum_%7Bj%5Cin%20J%7D%5Clambda_j%5Es&%5C%5C&&%5C%5C&x_i%5Cin%5C%7B0,1%5C%7D&%5Cforall%20i%5Cin%20I%5C%5C&y_%7Bij%7D%5Es%5Cgeq%200&%5Cforall%20i%5Cin%20I,%5Cforall%20j%5Cin%20J,%5Cforall%20s%5Cin%20S%5Cend%7Barray%7D%5Ctag%7B1%7D%5Cend%7Bequation%7D)

$
\begin{equation}
\begin{array}{rll}
 \min \quad & \sum_{i \in I} f_i x_i + \sum_{s \in S} \sum_{i \in I} \sum_{j \in J} \alpha^s q_{ij} y_{ij}^s & \\
 & &  \\
 \textrm{subject to} \quad & \sum_{i \in I} y_{ij}^s \geq \lambda_j^s & \forall j \in J, \forall s \in S \\
 & \sum_{j \in J} y_{ij}^s \leq k_i x_i & \forall i \in I, \forall s \in S \\
 & \sum_{i \in I} k_i x_i \geq \max_{s \in S} \sum_{j \in J} \lambda_j^s & \\
 & &  \\
 & x_i \in \{0, 1\} & \forall i \in I \\
 & y_{ij}^s \geq 0 & \forall i \in I, \forall j \in J, \forall s \in S
\end{array} \tag{1}
\end{equation}
$


## Pyomo Model
The model is encoded in the [pyomo_floc.py](src/pyomo_floc.py) file, which contains a function that instantiates the model given data generated based on the user's input.


## Generator
The [pyomo_gen_floc.py](src/pyomo_gen_floc.py) script generates data based on user input and creates a dictionary that is passed to the model instantiation function in [pyomo_floc.py](src/pyomo_floc.py) to instantiate the model.

Users can select the state they would like to work in, the number of facilities to consider for supplying products to customers, and the number of cities where customers demanding the products are located. Additionally, users can choose the number of different demand scenarios to incorporate into the model. Finally, they can specify the transportation costs and whether to scale or push the parameter values to IEEE representation limits.

The generator will then pick the most populous cities in the state for facilities (data is obtained by parsing the [uscities.csv](data/uscities.csv) file). For example, if the user chooses 7 facilities, the 7 most populous cities will be selected as potential distribution facilities. The remaining cities will be considered for customer locations. For example, if the user decides to have 70 customer locations, the 8th to the 77th most populous cities will be picked as customer locations.

Fixed costs for facilities are computed based on population: the more populated a city, the more expensive it is to open a facility. Variable costs, i.e., transportation costs, are computed based on the Haversine distance, which is then multiplied by the cost per distance input.

Demand and capacity parameters are also computed based on the population of the cities: the more populous a city, the higher its production capacity and the higher its demand for the product.

### Scaling
If scaling is turned on, the production and facility variables as well as the demand are multiplied by the scale factor. You can think of this as row scaling, i.e., multiplying each row by the scale factor.
