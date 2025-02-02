class Base:
    x = 1
    
    @classmethod
    def foo(cls):
        print(cls.x)
        
    def bar(self):
        print(self.x)

class Child1(Base):
    x = 2

class Child2(Base):
    x = 3

def foo(b: Base):
    print(Base.x)
    
Child2().foo()
Child2().bar()