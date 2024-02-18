import math
import matplotlib.pyplot as plt

# sort data in ascending order
def sort_data_ascending_order(data:list) -> list:
    data_float = [float(x) for x in data]
    print('data as intergers: ' +  str(data_float))
    data_float.sort()
    print('sorted-data: ' + str(data_float))
    return data_float

# find mean
def find_mean(data:list):
    data_sum = 0
    for data_val in data:
        data_sum += float(data_val)
    mean:float = data_sum / len(data)
    print('data_set mean: ' +  str(mean))
    return mean

# find median
def find_median(data:list):
    sorted_data = sort_data_ascending_order(data)
    # determine if observation length is even or odd
    if(len(sorted_data) % 2 == 0): # data set len is even
        # get middle two dataset values
        data_val_1_index:int = int(len(sorted_data) / 2)
        data_val_2_index:int = data_val_1_index - 1

        median: float = (float(sorted_data[data_val_1_index]) +  float(sorted_data[data_val_2_index])) / 2
        print('median avg vals indexes: 1-' + str(data_val_1_index) + ' 2-' + str(data_val_2_index))
        print('median: ' + str(median))
        return median
    else:
        median_index =int(((len(data) - 1) / 2))
        print('median index: ' +  str(median_index))
        median = sorted_data[median_index] # get data set middle value
        print('median: ' + str(median))
        return median
    

class mode_obj:
    def __init__(self) -> None:
        self.modes: list[mode_observation] = []

    def print_modes(self):
        if len(self.modes) > 0:
            for mode in self.modes:
                print('Mode: ' +  str(mode.mode_val) + ' Freq: ' + str(mode.mode_freq))
    
    def add_mode(self, mode_observation) -> None:
        self.modes.append(mode_observation)

    # this function checks if a number exist in a dataset and returns the freq
    def contains_val(self, test_val, data_set:list) -> int:
        val_freq:int = 0

        for val in data_set:
            if float(val) == float(test_val):
                val_freq += 1

        return val_freq
    
    # check if mode has been added already
    def does_observation_exist(self, val):
        if len(self.modes) != 0:
            for mode in self.modes:
                if float(mode.mode_val) == float(val):
                    return True
        
        return False


class mode_observation:
    def __init__(self,  mode_val, mode_freq) -> None:
        self.mode_val = mode_val
        self.mode_freq = mode_freq

    

# find Mode if any
def find_mode(data:list):
    sorted_data:list = sort_data_ascending_order(data)

    test_val_index:int = 0
    test_val:int = sorted_data[test_val_index]

    modes_ = mode_obj()

    while(test_val_index < len(sorted_data)):
        
        test_val_freq = modes_.contains_val(test_val, sorted_data)
        #print('mode ' + str(test_val) + ' Freq ' + str(test_val_freq))
        if test_val_freq > 1 and modes_.does_observation_exist(test_val) == False:
            mode_obs = mode_observation(test_val, test_val_freq)
            modes_.add_mode(mode_obs)

        test_val = sorted_data[test_val_index]
        test_val_index += 1
    
    modes_.print_modes()

    
# find sample variance
def find_sample_variance(data:list) -> float:
    sorted_data:list = sort_data_ascending_order(data)
    #get mean
    mean = find_mean(sorted_data)
    data_set_len = len(sorted_data)
    
    sample_variance: float = 0
    deviation_sum = 0
    for val in sorted_data:
        deviation_sum += float((float(val) - mean) ** 2)

    sample_variance = deviation_sum / (data_set_len - 1)
    print('sample variance: ' + str(sample_variance))
    return sample_variance

def find_population_variance(data:list) -> float:
    sorted_data:list = sort_data_ascending_order(data)
    #get mean
    mean = find_mean(sorted_data)
    data_set_len = len(sorted_data)

    population_variance: float = 0
    deviation_sum = 0
    for val in sorted_data:
        deviation_sum += float((float(val) - mean) ** 2)

    population_variance = deviation_sum / data_set_len
    print('population variance: ' + str(population_variance))
    return population_variance

def find_sample_standard_deviation(data:list) -> float:
    sample_standard_deviation: float = math.sqrt(find_sample_variance(data))
    print('sample standard deviation : ' + str(sample_standard_deviation))
    return sample_standard_deviation

def find_population_standard_deviation(data:list) -> float:
    population_standard_deviation: float = math.sqrt(find_population_variance(data))
    print('population standard deviation: ' + str(population_standard_deviation))
    return population_standard_deviation

def find_range(data:list) -> float:
    largest_val = float(max(data))
    smallest_val =  float(min(data))
    range: float = largest_val - smallest_val
    print('Range : ' + str(range) + '  -- largest val: ' + str(largest_val) + ' smallest val: ' + str(smallest_val)) 
    return range

def convert_values_to_float(data:list) -> list[float]:
    float_data_values: list[float] = [float(num) for num in data]
    print('data values as floats : ' + str(float_data_values))
    return float_data_values

def find_x_percentile(percentile: int,data:list) -> float:
    #Arrange data
    sorted_data: list =  sort_data_ascending_order(data)
    # find location of percentile in ordered data
    data_len = len(data)
    percentile_location:float = data_len * (percentile/100)
    # determine if location val is an integer
    if percentile_location.is_integer():
        # if int - get val in l position and val in l + 1 then avg the values for percentile val
        val_1:float = sorted_data[percentile_location]
        val_2:float = sorted_data[percentile_location + 1]
        percentile_location = (val_1 + val_2) / 2
        print(f'{percentile}th percentile value is {sorted_data[percentile_location - 1]} at location {percentile_location}')
        return sorted_data[percentile_location - 1]
    else:
        # if not int - round up to closest int, then return val in l location of the orderd data
        percentile_location = percentile_location.__ceil__()
        print(f'{percentile}th percentile value is {sorted_data[percentile_location - 1]} at location {percentile_location}')
        return sorted_data[percentile_location - 1]
    
def find_interquartile_range(data:list) -> float:
    # Interquartile range = Q1 - Q3
    Q1: float = find_x_percentile(25,data)
    Q3: float = find_x_percentile(75,data)
    interquartile_range: float = Q3 - Q1
    print(f'Interquartile Range : {interquartile_range}')
    return interquartile_range


def find_percentile_of_x(data_val:float, data:list) -> int:
    # order data
    sorted_data = sort_data_ascending_order(data)
    vals_less_than_ref = sum(1 for val in sorted_data if val <= data_val)
    data_set_len = len(data_set)
    percentile = (vals_less_than_ref / data_set_len) * 100
    print(f'Percentile for value {data_val} - Percentile: {percentile}')
    return int(percentile)

def find_correlation_coefficient_relationship(data_A:list, data_B:list) -> float:
    '''
    Correlation Coefficient Computation Formula:
    r = ((n∑xᵢyᵢ) - (∑xᵢ)(∑yᵢ)) / √((n∑xᵢ²) - (∑xᵢ)²) * √((n∑yᵢ²) - (∑yᵢ)²)
    '''
    data_A_f:list[float] = convert_values_to_float(data_A)
    data_B_f:list[float] = convert_values_to_float(data_B)

    if data_A_f.__len__() != data_B_f.__len__():
        print('the two data sets do not Match')
        print(f'data set A : {data_A_f}')
        print(f'data set B : {data_B_f}')
        print('Please Re-Enter Datasets...')
        data_A = input('DataSet_A (csv): ').split(',')
        data_B = input('DataSet_B (csv):').split(',')
        data_A_f = convert_values_to_float(data_A)
        data_B_f = convert_values_to_float(data_B)
    
    n: int = data_A_f.__len__() # number of data values
    # ∑xᵢyᵢ
    sum_of_products_Xi_Yi: float = 0
    i = 0
    while i < n:
        sum_of_products_Xi_Yi += data_A_f[i] * data_B_f[i]
        i += 1
    print(f'∑xᵢyᵢ : {sum_of_products_Xi_Yi}')
    # ∑xᵢ
    sum_of_Xi: float = 0
    for val in data_A_f:
        sum_of_Xi += val
    print(f'∑xᵢ : {sum_of_Xi}')
    # ∑Yᵢ
    sum_of_Yi: float = 0
    for val in data_B_f:
        sum_of_Yi += val
    print(f'∑Yᵢ : {sum_of_Yi}')
    # ∑Xᵢ²
    sum_of_Xi_sqrd: float = 0
    for val in data_A_f:
        sum_of_Xi_sqrd += val * val
    print(f'∑Xᵢ² : {sum_of_Xi_sqrd}')
    # ∑Yᵢ²
    sum_of_Yi_sqrd: float = 0
    for val in data_B_f:
        sum_of_Yi_sqrd += val * val
    print(f'∑Yᵢ² : {sum_of_Yi_sqrd}')
    #Relation
    r: float =  ((n * sum_of_products_Xi_Yi) - ((sum_of_Xi) * (sum_of_Yi))) / (math.sqrt((n * sum_of_Xi_sqrd) - (sum_of_Xi * sum_of_Xi)) * math.sqrt((n * sum_of_Yi_sqrd) - (sum_of_Yi * sum_of_Yi)))

    print(f'Relation Coeffiecient: {r}')
    return r


def fit_linear_model(data_A_f:list[float], data_B_f:list[float]) -> None:
    '''
    Linear Regression prediction Model (Stats Format): y = b₀ + b₁x
    Measure of how close data is, will be used for finding best fitting line. : Sum of Squared Errors (SSE)=
    SSE = ∑errorᵢ² = ∑(yᵢ - ŷᵢ)² = ∑(yᵢ - (b₀ + b₁x))²
    Error = yᵢ - ŷᵢ
    Sum_Error = ∑error

    Finding Least Squares Line (finding prediction slope and y-intercept):
    Slope :  b₁ = (n∑xᵢyᵢ - ∑xᵢ∑yᵢ) / (n∑xᵢ² - (∑xᵢ)²)
    y-intercept : b₀ = ȳ - b₁x_mean =  (1 / n)(∑yᵢ - b₁∑xᵢ)
    note: the slope coefficient needs to be calculated before the y-intercept

    steps:
    * Find slope
    * Find y-intercept
    * calculate prediected Y
    * calculate Error
    * calculate Squared Error
    '''
    # check if datasets are equal
    if data_A_f.__len__() != data_B_f.__len__():
        print('the two data sets do not Match')
        print(f'data set A : {data_A_f}')
        print(f'data set B : {data_B_f}')
        print('Please Re-Enter Datasets...')
        data_A = input('DataSet_A (csv): ').split(',')
        data_B = input('DataSet_B (csv):').split(',')
        data_A_f = convert_values_to_float(data_A)
        data_B_f = convert_values_to_float(data_B)

    # observed X: data_A and observered Y: data_B
    #Find Slope:  b₁ = (n∑xᵢyᵢ - ∑xᵢ∑yᵢ) / (n∑xᵢ² - (∑xᵢ)²)
    n = data_A_f.__len__()
    print(f'n = {n}')
    #∑xᵢyᵢ
    sum_of_Xi_Yi:float = 0
    i:int = 0
    while i < data_A_f.__len__():
        sum_of_Xi_Yi += data_A_f[i]*data_B_f[i]
        i += 1
    print(f'∑xᵢyᵢ = {sum_of_Xi_Yi}')
    #∑xᵢ∑yᵢ
    product_of_sum_of_Xi_sum_of_Yi:float = 0
    i = 0
    sum_of_Xi:float = 0
    sum_of_Yi:float = 0
    while i < data_A_f.__len__():
        sum_of_Xi += data_A_f[i]
        sum_of_Yi += data_B_f[i]
        i += 1
    product_of_sum_of_Xi_sum_of_Yi = sum_of_Xi * sum_of_Yi
    print(f'∑xᵢ∑yᵢ =  {product_of_sum_of_Xi_sum_of_Yi}')
    #∑xᵢ
    sum_of_Xi = 0
    i = 0
    while i < data_A_f.__len__():
        sum_of_Xi += data_A_f[i]
        i += 1
    print(f'∑xᵢ = {sum_of_Xi}')
    #∑xᵢ²
    sum_of_Xi_sqrd:float = 0
    i = 0
    while i < data_A_f.__len__():
        sum_of_Xi_sqrd += data_A_f[i] * data_A_f[i]
        i += 1
    print(f'∑xᵢ² = {sum_of_Xi_sqrd}')
        

    slope:float = ((n * sum_of_Xi_Yi) - product_of_sum_of_Xi_sum_of_Yi) / ((n * (sum_of_Xi_sqrd)) - (sum_of_Xi * sum_of_Xi))
    print(f'slope = {slope}')
    # Find Y-intercept: b₀ = ȳ - b₁x_mean =  (1 / n)(∑yᵢ - b₁∑xᵢ)
    #∑yᵢ
    sum_of_Yi:float = 0
    i = 0
    while i < data_B_f.__len__():
        sum_of_Yi += data_B_f[i]
        i += 1
    print(f'∑yᵢ = {sum_of_Yi}')
    y_intercept:float =  (1 / n)* (sum_of_Yi - (slope * sum_of_Xi))
    
    print(f'Computed Prediction Model: Y = {y_intercept} + {slope}(x)')

    # Calculate predicted Y Values: y = b₀ + b₁x
    predicted_y_values:list[float] = []
    for val in data_A_f:
        predicted_val:float = y_intercept + (slope * val)
        predicted_y_values.append(predicted_val)
    print(f'predicted Y values: {predicted_y_values}')

    # Calculate Error: error = yᵢ - ŷᵢ
    errors:list[float] = []
    sum_of_errors:float  = 0
    i = 0
    while i < predicted_y_values.__len__():
        error:float =  data_B_f[i] - predicted_y_values[i]
        sum_of_errors += error
        errors.append(error)
        i += 1
    print(f'sum of errors: {sum_of_errors} \nerrors : {errors}')

    # Calculate Error Squared
    errors_sqrd:list[float] = []
    sum_of_errors_sqrd:float = 0
    for val in errors:
        error_sqrd:float = val * val
        sum_of_errors_sqrd += error_sqrd
        errors_sqrd.append(error_sqrd)
    print(f'sum of errors²: {sum_of_errors_sqrd} \nerrors² {errors_sqrd}')

    #Create Visual Plot
    plt.plot(data_A_f,data_B_f,'o', markersize=10, label='Data Points')

    #create model line
    plt.plot(data_A_f, predicted_y_values, color='red', label='Regression Line')

    plt.xlabel('X-axis Dataset A')
    plt.ylabel('Y-axis Dataset B')
    plt.title('Fit Linear Model')

    plt.legend()

    plt.show()
    


    


    


# data_set_func_menu
def data_set_funcs_menu(data_set):
    print()
    print()
    tool_selection = input("Enter Desired Tool (type tool name) -- 'new_data_set','sort-ascending', 'mean', 'median', 'mode', 'population_variance', 'population_standard_deviation', 'sample_variance', 'sample_standard_deviation', 'range', 'convert_to_float', 'x-percentile', 'interquartile_range', 'percentile_of_val', 'correlation_coefficient_r', 'fit_linear_model' : ")
    if(tool_selection != None):
        if tool_selection == 'sort-ascending':
            print("Sort Data Ascending Order")
            sort_data_ascending_order(data_set)
            data_set_funcs_menu(data_set)
        if tool_selection == 'mean':
            print("Finding Mean of Data Set: ")
            find_mean(data_set)
            data_set_funcs_menu(data_set)
        if tool_selection == 'median':
            print("Finding Mean of Data Set:")
            find_median(data_set)
            data_set_funcs_menu(data_set)
        if tool_selection == 'mode':
            print("Finding Mode of Data Set:")
            find_mode(data_set)
            data_set_funcs_menu(data_set)
        if tool_selection == 'sample_variance':
            print('Finding sample_variance of Data Set')
            find_sample_variance(data_set)
            data_set_funcs_menu(data_set)
        if tool_selection == 'new_data_set':
            data_set = input("Enter new data values :")
            if (data_set != None):
                data_set = data_set.split(',')
                print('Entered data-set : ' + str(data_set))
                data_set_funcs_menu(data_set)
        if tool_selection == 'sample_standard_deviation':
            print('Finding sample standard deviation')
            find_sample_standard_deviation(data_set)
            data_set_funcs_menu(data_set)
        if tool_selection == 'range':
            print('Finding Range of data set')

            find_range(convert_values_to_float(data_set))
            data_set_funcs_menu(data_set)
        if tool_selection == 'convert_to_float':
            print('Converting Data set to float')
            data_set_funcs_menu(convert_values_to_float(data_set))
        if tool_selection ==  'x-percentile':
            percentile = input('Enter Percentile to find for data set (Integer) :')
            find_x_percentile(int(percentile), data_set)
            data_set_funcs_menu(data_set)
        if tool_selection == 'interquartile_range':
            find_interquartile_range(data_set)
            data_set_funcs_menu(data_set)
        if tool_selection == 'percentile_of_val':
            compare_val = input('Enter value to find percentile of (float):')
            find_percentile_of_x(float(compare_val), data_set)
            data_set_funcs_menu(data_set)
        if tool_selection == 'population_variance':
            print('Finding Population Sample Variance')
            find_population_variance(data_set)
            data_set_funcs_menu(data_set)
        if tool_selection == 'population_standard_deviation':
            print('Finding Population Standard Deviation')
            find_population_standard_deviation(data_set)
            data_set_funcs_menu(data_set)
        if tool_selection == 'correlation_coefficient_r':
            print('Finding the Correlation Coefficient Relationship:')
            data_set_b = input('Enter the comparison data set values (csv): ').split(',')
            find_correlation_coefficient_relationship(data_set, data_set_b)
            data_set_funcs_menu(data_set)
        if tool_selection == 'fit_linear_model':
            print('Fit linear model for dataset X and dataset Y, where y is the prediction dataset and X is the independent variable dataset:')
            data_set_y = input('Enter Prediction DataSet (csv):').split(',')
            fit_linear_model(convert_values_to_float(data_set), convert_values_to_float(data_set_y))
            data_set_funcs_menu(data_set)

if __name__ == "__main__":
    data_set = input("Enter data comma seperated values: ")
    if (data_set != None):
        data_set = data_set.split(',')
    
        print('Entered data-set : ' + str(data_set))    
        data_set_funcs_menu(data_set)
