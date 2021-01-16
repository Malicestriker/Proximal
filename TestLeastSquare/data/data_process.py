import numpy as np

'''
f = 'cpusmall_raw.txt'
f = open(f,'r')

output = 'cpusmall.txt'
output = open(output,'w')

A = []
b = [] 
for line in f:
    data = line.split()
    row = [].copy()
    b.append(float(data[0]) )
    for s in data[1:]:
        tmp = s.split(':')
        row.append(float(tmp[1]))
    
    row.append(1)
    A.append(row.copy())

m = len(A)
n = len(A[0])
print("{0} {1}".format(m-1000,n), file = output)

for i in range(m-1000):
    for j in range(n):
        print("%f"%A[i][j], file = output, end = ' ')
    print('',file = output)

print("{0} {1}".format(m-1000,1), file = output)
for i in range(m):
    print("%f"%b[i], file = output, end = ' ')

test = 'cpusmall_test.txt'
test = open(test,'w')
print("{0} {1}".format(1000,n), file = test)
for i in range(m-1000,m):
    for j in range(n):
        print("%f"%A[i][j], file = test, end = ' ')
    print('',file = test)

print("{0} {1}".format(1000,1), file = test)
for i in range(m-1000,m):
    print("%f"%b[i], file = test, end = ' ')

At = np.array(A[:-1000])
bt = np.array(b[:-1000])
print(np.shape(At),np.shape(bt))
print( np.linalg.lstsq(At,bt) )
'''

f = 'a1a_raw.txt'
f = open(f,'r')


n = 1605

A = np.zeros([n,121])
b = np.zeros([n,1])
for i,line in enumerate(f):
    data = line.split()
    b[i] = float( data[0] )
    for s in data[1:]:
        tmp = s.split(':')
        A[i][int(tmp[0])] = float(tmp[1])
    A[i][-1] = 1
    

output = 'a1a_train.txt'
output = open(output,'w')

m = len(A[0])
nt = 900
print(m,n)
print("%d %d"%(nt,m),file = output)
for i in range(nt):
    for j in range(m):
        print("%f"%A[i][j],file = output, end = ' ')
    print('',file = output)

print("%d %d"%(nt,1),file = output)
for i in range(nt):
    print("%d"%b[i],file = output)

output = 'a1a_test.txt'
output = open(output,'w')

print("%d %d"%(n-nt,m),file = output)
for i in range(nt,n):
    for j in range(m):
        print("%f"%A[i][j],file = output, end = ' ')
    print('',file = output)

print("%d %d"%(n-nt,1),file = output)
for i in range(nt,n):
    print("%d"%b[i], file = output)