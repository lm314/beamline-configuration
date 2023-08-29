import ast
import yaml
import numpy as np

# Try to import particle_accelerator_utilities for calculation of relativistic 
# quantities.
try:
    import pyPartAnalysis.particle_accelerator_utilities as pau
except ImportError:
    pass

class ListDict(dict):
    # each value is a list of equal length or None
    # iterating over the ListDict iterates over the values in the list 
    # returning a dictionary with all the keys
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_lengths()

    def __setitem__(self, key, value):
        if value is not None and not isinstance(value, (list, int, float)):
            raise TypeError("Value must be None, a scalar or a list")
        super().__setitem__(key, value)
        self._check_lengths()

    def _check_lengths(self):
        lengths = [len(v) for v in self.values() if isinstance(v, list)]
        if len(set(lengths)) > 1:
            raise ValueError("All lists must have the same length")

    def __iter__(self):
        self._check_lengths()
        if all(isinstance(v, (int, float)) for v in self.values()):
            yield self
            return
        self._current_index = 0
        try:
            self._max_index = len(next(iter(v for v in self.values() if isinstance(v, list))))
        except StopIteration:
            self._max_index = 0
        return self

    def __next__(self):
        if self._current_index >= self._max_index:
            raise StopIteration
        result = {k: (v[self._current_index] if isinstance(v, list) else v) for k, v in self.items()}
        self._current_index += 1
        return result
        
class BeamlineConfiguration:
    # Processes a settings yaml file to create an input dictionary for use with impact_input
    def __init__(self,settings):
        self.settings = settings
        
        # holds the values created from settings 
        self.__input_dict = self.__create_dict()
        
        #contains the values after transforming __input_dict
        self.__output_dict = self.__create_ListDict()
        
    def gen(self,matched_lengths=False):
        # generate dictionary from settings
        # matched_lengths means all variables have the same number 
        # of values and that we do not want every possible combination in the output dictionary
        self.__process_initial_values()
        if not matched_lengths:
            self.__populate_initial_values()
        for key,val in self.settings.items():
            self.__transform_initial_values(key,val)
        
        return self.__output_dict      

    @staticmethod
    def split(d):
        # splits __output_dict into dictionary based off of prefixes of the 
        # variable names, e.g. {'name1__a': 1, 'name2__b': 2, 'c': 3} becomes
        # {'name1': {'a': 1}, 'name2': {'b': 2}, 'original': {'c': 3}}
        # if splitting does occur, the value for each key is a ListDict
        result = {}
        for k, v in d.items():
            if '__' in k:
                prefix, suffix = k.split('__')
                if prefix in result:
                    result[prefix][suffix] = v
                else:
                    result[prefix] = {suffix: v}
            else:
                if 'original' in result:
                    result['original'][k] = v
                else:
                    result['original'] = {k: v}
        if not result:
            return d
        else:
            for key,val in result.items():
                result[key] = ListDict(zip(val.keys(),val.values()))
            return result
    
    def __process_initial_values(self):
        # for processing 'input' key from settings
        # generates values for input_dict from self.settings
        
        # all options for generating values
        switcher = {
        frozenset([]): lambda x: None,
        frozenset(['value']): lambda x: x['value'],
        frozenset(['min','max','number_steps']): lambda x: np.linspace(x['min'],x['max'],x['number_steps']).tolist(),
        frozenset(['min','max','step_size']): lambda x: np.arange(x['min'],x['max']+x['step_size']/2,x['step_size']).tolist(),
        }
        
        for key,val in self.settings.items():
            input_keys = frozenset(val.get('input',{}).keys())
            self.__input_dict[key] = switcher[input_keys](val.get('input',None))
            
    def __populate_initial_values(self):
        # get values for all independent variables and make every possible combination of variables
        ind_vars = self.__get_independent_var()
        temp_vars = [np.array(self.__input_dict[ind_var]) for ind_var in ind_vars]
        
        temp_arr = self.__makeInputs(*temp_vars)

        # redistribute new values with all combination to their original dict entries
        # if values for inputs are not scalars, determine how to make all outputs have the same output length
        for ind_var,arr in zip(ind_vars,temp_arr.T):
            temp = arr.tolist()
            if len(temp) == 1:
                temp = temp[0]
            self.__input_dict[ind_var] = temp
            
    def __transform_initial_values(self,key,val):
        # transforms the initial values in the input_dict using the 'output' key of settings
        
        switcher = {
        frozenset([]): lambda x: x,
        frozenset(['function']): lambda x: self.__eval_function(x['function'])
        }
        
        input_keys = frozenset(val.get('output',{}).keys())
        
        switcher_input = self.__input_dict.get(key) if self.settings.get(key).get('output') is None else self.settings.get(key).get('output')
        
        self.__output_dict[key] = switcher[input_keys](switcher_input)
        
    def __process_function_string(self,formula):
        # Creates new string that can be evaluated by python
        node_list = [
            node for node in ast.walk(ast.parse(formula)) 
            if isinstance(node, ast.Name)
        ]
        
        # possible name of the variables in the formula
        names = [node.id for node in node_list]
        
        # filter out only candidates that appear in settings
        names = [x for x in names if x in self.settings.keys()]
        
        # function must depend on variables defined in settings, otherwise names is empty
        assert names != None
            
        #all variables need to be independent
        # assert all([self.__check_variable_independent(x) for x in names])
        
        # find the pairs of successive breakpoints in the formula based off of the variables
        temp = sorted(list(set(sorted([0]+[node.col_offset for node in node_list]+[node.end_col_offset for node in node_list]+[len(formula)]))))
        
        pairs = [(first, second) for first, second in zip(temp, temp[1:])]
        
        new_form = ''
        
        for pair0,pair1 in pairs:
            if formula[pair0:pair1] in names:
                #if pair of breakpoints is for a variable, return the variable value
                # if input did not exist in settings, use the calculated value from output                
                if self.__input_dict[formula[pair0:pair1]] is not None:
                    value = self.__input_dict[formula[pair0:pair1]]
                else:
                    self.__transform_initial_values(formula[pair0:pair1],self.settings[formula[pair0:pair1]])
                    value = self.__output_dict[formula[pair0:pair1]]
                    
                new_form += f"np.array({value})"
            else:
                new_form += formula[pair0:pair1]   
        return new_form
    
    def __eval_function(self,formula):
        # uses eval, which can be unsafe as users can inject malicious code
        new_function_str = self.__process_function_string(formula) 
        return eval(new_function_str).tolist()
    
    def __check_variable_independent(self,var):
        ind_vars = self.__get_independent_var()
        return var in ind_vars
    
    def __get_independent_var(self):
        # gets variables that calculated using their own input value
        return [key for key in self.settings.keys() if 'input' in self.settings[key]]
    
    def __makeInputs(self,*args):
        # given lists of inputs, outputs 2d array with each combination of the lists
        grid_mats = np.meshgrid(*np.array([*args],dtype=object))
        temp = []
        for grid in grid_mats:
            temp.append(grid.ravel())
        return np.array(temp).T
    
    def __create_dict(self):
        # makes an empty dictionary with values None with the keys from settings. 
        key_list = self.settings.keys()
        d = {}
        d.__init__(zip(key_list, [None]*len(key_list)))
        return d
    
    def __create_ListDict(self):
        # makes an empty dictionary with values None with the keys from settings. 
        key_list = self.settings.keys()
        d = ListDict({})
        d.__init__(zip(key_list, [None]*len(key_list)))
        return d
    
def main():
    # run test cases
    for i in np.arange(1,9):
        print(i)
        with open(f'input_{i}.yaml', 'r') as file:
            input_dict = ListDict(yaml.safe_load(file))
            # input_dict = yaml.safe_load(file)

        with open(f'output_{i}.yaml', 'r') as file:
            output_dict_test = ListDict(yaml.safe_load(file))
            # output_dict_test = yaml.safe_load(file)

        a=BeamlineConfiguration(input_dict)

        output_dict = a.gen()
        print(output_dict==output_dict_test)
        
if __name__ == '__main__':
    main()
