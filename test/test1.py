

def dec(func, y='456'):
    def func1(x):
        x = 'ABC'+ x + y
        return func(x)
    return func1


@dec
def abcd(str1):
    str1 = str1+'123'
    print (str1)
    return str1

@dec
def efgh(str1):
    str1 = str1+'123'
    print(str1)
    return str1

#abcd('abcd')
#efgh('efgh')

def deca(func, args1=1):
    def func1(a, b):
        if b == 0:
            print ('Can Not divide')
            return 0
        if b == args1:
            print ('Can divide')
            return a 
        else:
            return func(a,b)
    
    return func1

@deca
def divv(a,b):
    print ("divide {} and {}".format(a,b))
    return a/b

print(divv(20,1))




