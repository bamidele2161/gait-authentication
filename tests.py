import unittest as ut
import codeHw0

class HW0TestCase(ut.TestCase):

    def test_hello(self):
        netid = codeHw0.netid
        str = codeHw0.hello()
        
        self.assertTrue(str[:6] == "Hello ")
        self.assertTrue(str[6:] == netid)

def do_tests():

    test_suite = ut.TestLoader().loadTestsFromTestCase(HW0TestCase)
    results = ut.TextTestRunner(verbosity=2).run(test_suite)
    total, errors, fails = results.testsRun, len(results.errors), len(results.failures)
    return total, errors, fails


if __name__ == "__main__":    
  
    total, errors, fails = do_tests()
    print("Score = %d out of %d (%d errors, %d failed assertions)" % (
        total - (errors + fails), total, errors, fails))
