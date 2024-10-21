class BBB:
    def method_in_bbb(self):
        print("This is a method in BBB.")

class AAA(BBB):
    pass

aaa_instance = AAA()
aaa_instance.method_in_bbb()
