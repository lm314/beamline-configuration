import os
import yaml
from parameterized import parameterized
import unittest

from beamline_configuration import BeamlineConfiguration

class TestBeamlineConfiguration(unittest.TestCase):
    @parameterized.expand(list(range(1,10)))
    def test_gen(self, case_number):
        with open(os.path.join(__location__,f'case_{case_number}.yaml'), 'r') as file:
            input_dict,output_dict_test = (x for x in yaml.safe_load_all(file))
            
        beamline_config = BeamlineConfiguration(input_dict)
        output_dict = beamline_config.gen()
        
        self.assertDictEqual(output_dict_test, output_dict)
        

if __name__ == '__main__':
    unittest.main()