import unittest
from typing import List


def payments(values: List[float], func) -> List[float]:
    pay = []
    c = func(values)
    #
    for i in range(len(values)):
        c1 = func(values)
        print("c1: ",c1)
        temp_values = values
        if c[i]:
            print("c: ",c)
            # decrease 0.01 from the price as long as he is still chose
            while c1[i]:
                temp_values[i] -= 0.01
                c1 = func(temp_values)
            pay.append(temp_values[i] + 0.01)
        else:
            # if index not choose the price he need to pay is 0
            pay.append(0)
    pay = ["%.2f" % elem for elem in pay]
    return pay


# choose all value that biggest then 10
def more_then_ten(values: List[float]) -> List[bool]:
    choices = []
    for val in values:
        #print("val: ",val)
        choices.append(True if val >= 10 else False)
    return choices


# choose the biggest value
def greedy_a(values: List[float]) -> List[bool]:
    choices = []
    max_value = max(values)
    #print(max_value)
    for val in values:
        choices.append(True if val >= max_value else False)
    return choices


class MyTestCase(unittest.TestCase):
    def test_more_then_ten(self):
        l1 = [12, 5, 7, 10, 9, 13, 9.95, 10.1]
        print(l1)
        answer = payments(l1, more_then_ten)
        print(answer)
        self.assertAlmostEqual(float(answer[0]), 10, delta=0.0001)
        self.assertAlmostEqual(float(answer[3]), 10, delta=0.0001)
        self.assertAlmostEqual(float(answer[5]), 10, delta=0.0001)
        self.assertAlmostEqual(float(answer[7]), 10, delta=0.0001)
    '''
    def test_greedy_a(self):
        print("test_greedy_a")
        l1 = [12, 5, 7, 10, 9, 13, 9.95, 10.1]
        #print(l1)
        answer = payments(l1, greedy_a)
        #print(answer)
        self.assertAlmostEqual(float(answer[5]), 12, delta=0.0001)
        l2 = [45, 45.2, 44.8, 52, 51.85, 51.87, 51.8]
        #print(l2)
        answer1 = payments(l2, greedy_a)
        #print(answer1)
    '''

if __name__ == '__main__':
    unittest.main()