import math as m

# calculate the population standard deviation
def stdev_p(data):
    if len(data)<2 or len(data)==[]:
        raise ValueError('At least two data points needed')
    n=len(data)
    z=0
    for y in data:
        z+=y
    m=z/n
    
    s=0
    for x in data:
        s+=(x-m)**2
    result = (s/n)**.5 # your code goes here
    return result

# calculate the sample standard deviation
def stdev_s(data):
    if len(data)<2 or len(data)==[]:
        raise ValueError('At least two data points needed')
    n=len(data)
    z=0
    for y in data:
        z+=y
    m=z/n
    
    s=0
    for x in data:
        s+=(x-m)**2
    result = (s/(n-1))**.5 # your code goes here
    return result

if __name__ == "__main__":
    test = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    print "the population stdev is", stdev_p(test)
    print "the sample stdev is", stdev_s(test)

