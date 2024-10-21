class A:
    def __init__(self):
        print("A init")

class B:
    def __init__(self):
        print("B init")

class C(A, B):
    def __init__(self):
        #super(C, self).__init__()  # 明确指定从 A 和 B 的继承顺序中查找父类的构造方法
        super(C,self).__init__()  # 明确指定从 A 和 B 的继承顺序中查找父类的构造方法
#        super(A,self).__init__()  # 明确指定从 A 和 B 的继承顺序中查找父类的构造方法
        print("C init")


c = C()
