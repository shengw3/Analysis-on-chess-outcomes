# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 13:32:15 2019

@author: Kevin
"""

# Enter 1550 and 1450 for test cases

import unittest
from project import *

class LogisticTestCase(unittest.TestCase):
    
    def test_probwin_1(self):
        self.assertNotEqual(chessprobwin(1550,1450), 0.5)
        #Making sure our probabilities match how they'd be calculated on pencil an paper
       
    def test_probwin_2(self):
        self.assertEqual(chessprobwin(1550,1450), 0.6014597959120163)
    
    def test_probdraw_1(self):
        self.assertEqual(chessprobdraw(1550,1450), 0.04621495230703922)
    
    def test_probdraw_2(self):
        self.assertNotEqual(chessprobdraw(1550,1450), 0.04)
        
    def test_increment_to_starting(self):
        self.assertEqual(increment_to_starting(['1+2','3+4']), [1,3])
        
    def test_increment_to_increment(self):
        self.assertEqual(increment_to_increment(['1+2','3+4']),[2,4])
       
if __name__ == '__main__':
    unittest.main() 